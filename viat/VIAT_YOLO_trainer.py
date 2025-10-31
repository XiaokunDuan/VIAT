"""
VIAT框架适配YOLO - 场景级旋转鲁棒训练
Scene-level Rotation Robustness Training with YOLO
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import yaml

from GMVFool_YOLO_v1 import GMVFoolYOLO
from gs3lam_interface import GS3LAMRenderer
from ultralytics import YOLO


class VIATYOLOTrainer:
    """
    VIAT训练器 - YOLO版本
    核心思想：通过对抗视角训练提升YOLO在场景旋转下的鲁棒性
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 初始化YOLO模型
        self.init_yolo_model()

        # 2. 初始化GS3LAM渲染器
        self.init_gs3lam_renderers()

        # 3. 初始化GMVFool
        self.init_gmvfool()

        # 4. 存储每个场景的对抗视角分布
        self.adversarial_distributions = {}

    def init_yolo_model(self):
        """
        初始化YOLO模型
        支持：YOLOv8, YOLOv9, YOLO-World等
        """
        print(f"Initializing YOLO model: {self.config['yolo_model']}")

        # 加载预训练模型
        self.yolo = YOLO(self.config['yolo_model'])  # 如 'yolov8x.pt'

        # 如果需要微调到特定数据集（如Replica的类别）
        if self.config.get('finetune_on_dataset'):
            self.finetune_yolo_on_scenes()

        self.yolo.to(self.device)
        print(f"✅ YOLO model loaded: {self.yolo.model}")

    def finetune_yolo_on_scenes(self):
        """
        在场景数据上微调YOLO
        将COCO类别映射到场景特定类别（如Replica的物体）
        """
        print("Fine-tuning YOLO on scene dataset...")

        # 准备训练数据
        # TODO: 将GS3LAM场景的语义标签转换为YOLO格式
        # 格式: class_id, x_center, y_center, width, height (归一化)

        train_config = {
            'data': self.config['yolo_data_yaml'],  # 数据配置文件
            'epochs': self.config.get('finetune_epochs', 50),
            'imgsz': self.config.get('image_size', 640),
            'batch': self.config.get('batch_size', 16),
            'device': self.device
        }

        # 执行训练
        results = self.yolo.train(**train_config)
        print(f"✅ YOLO fine-tuning completed")

    def init_gs3lam_renderers(self):
        """
        初始化所有场景的GS3LAM渲染器
        """
        self.renderers = {}
        scene_dirs = Path(self.config['scene_data_dir']).glob('*')

        for scene_dir in scene_dirs:
            if scene_dir.is_dir():
                scene_id = scene_dir.name
                print(f"Loading GS3LAM for scene: {scene_id}")

                renderer = GS3LAMRenderer(
                    scene_path=str(scene_dir),
                    checkpoint_path=self.config.get('gs3lam_checkpoint'),
                    image_size=tuple(self.config['image_size'])
                )

                self.renderers[scene_id] = renderer

        print(f"✅ Loaded {len(self.renderers)} GS3LAM renderers")

    def init_gmvfool(self):
        """初始化GMVFool攻击器（每个场景一个）"""
        self.gmvfools = {}

        for scene_id, renderer in self.renderers.items():
            gmvfool = GMVFoolYOLO(
                yolo_model_path=self.config['yolo_model'],
                gs3lam_renderer=renderer,
                num_components=self.config['num_gaussian_components'],
                lambda_entropy=self.config['lambda_entropy']
            )
            self.gmvfools[scene_id] = gmvfool

        print(f"✅ Initialized GMVFool for {len(self.gmvfools)} scenes")

    def train(self):
        """
        VIAT主训练循环
        遵循原始VIAT的两阶段策略
        """
        print("=" * 50)
        print("Starting VIAT-YOLO Training")
        print("=" * 50)

        # 阶段1: 初始化所有场景的对抗视角分布（粗优化）
        self.phase1_initialize_distributions()

        # 阶段2: 迭代优化分布 + 训练YOLO
        self.phase2_iterative_training()

        # 保存最终模型
        self.save_model()

        print("✅ VIAT-YOLO Training Completed!")

    def phase1_initialize_distributions(self):
        """
        阶段1: 为所有场景初始化对抗视角分布
        这个阶段比较耗时，但只需要执行一次
        """
        print("\n" + "="*50)
        print("Phase 1: Initializing Adversarial Viewpoint Distributions")
        print("="*50)

        for scene_id, gmvfool in tqdm(self.gmvfools.items(), desc="Scenes"):
            # 获取场景中的目标物体类别
            target_objects = self.get_scene_target_objects(scene_id)

            # 优化对抗视角分布（粗优化，较多迭代次数）
            print(f"\nOptimizing distribution for scene: {scene_id}")
            optimized_dist = gmvfool.optimize_adversarial_viewpoints(
                scene_id=scene_id,
                target_objects=target_objects,
                num_iterations=self.config['phase1_iterations'],  # 如 50
                learning_rate=self.config['learning_rate'],
                num_samples=self.config['num_samples']
            )

            # 保存分布参数
            self.adversarial_distributions[scene_id] = optimized_dist

            # 可视化并保存一些对抗视角样本
            self.visualize_adversarial_views(scene_id, gmvfool, num_samples=10)

        print("\n✅ Phase 1 completed: All distributions initialized")

    def phase2_iterative_training(self):
        """
        阶段2: 迭代训练
        每个epoch:
        1. 随机选择部分场景，精细优化对抗分布（stochastic update）
        2. 从分布采样对抗视角，渲染图像
        3. 在对抗图像 + 自然图像上训练YOLO
        """
        print("\n" + "="*50)
        print("Phase 2: Iterative Adversarial Training")
        print("="*50)

        num_epochs = self.config['num_epochs']
        scenes_per_epoch = self.config.get('scenes_per_epoch', 5)  # 每个epoch更新的场景数

        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")

            # 1. 随机选择部分场景进行分布更新（stochastic update strategy）
            selected_scenes = np.random.choice(
                list(self.gmvfools.keys()),
                size=min(scenes_per_epoch, len(self.gmvfools)),
                replace=False
            )

            for scene_id in selected_scenes:
                # 精细优化分布（较少迭代次数）
                target_objects = self.get_scene_target_objects(scene_id)
                updated_dist = self.gmvfools[scene_id].optimize_adversarial_viewpoints(
                    scene_id=scene_id,
                    target_objects=target_objects,
                    num_iterations=self.config['phase2_iterations'],  # 如 10
                    learning_rate=self.config['learning_rate'] * 0.5,  # 降低学习率
                    num_samples=self.config['num_samples']
                )
                self.adversarial_distributions[scene_id] = updated_dist

            # 2. 构建训练batch（自然视角 + 对抗视角）
            train_images, train_labels = self.build_training_batch(epoch)

            # 3. 训练YOLO一个epoch
            self.train_yolo_one_epoch(train_images, train_labels, epoch)

            # 4. 验证
            if (epoch + 1) % self.config['eval_frequency'] == 0:
                self.evaluate(epoch)

    def build_training_batch(self, epoch):
        """
        构建训练batch
        包含：自然视角图像 + 对抗视角图像
        """
        train_images = []
        train_labels = []

        # 自然视角图像（从原始数据集）
        natural_images, natural_labels = self.load_natural_images(
            num_samples=self.config['natural_samples_per_epoch']
        )
        train_images.extend(natural_images)
        train_labels.extend(natural_labels)

        # 对抗视角图像（从分布采样并渲染）
        adversarial_images, adversarial_labels = self.sample_adversarial_images(
            num_samples=self.config['adversarial_samples_per_epoch']
        )
        train_images.extend(adversarial_images)
        train_labels.extend(adversarial_labels)

        print(f"Built training batch: {len(train_images)} images "
              f"({len(natural_images)} natural + {len(adversarial_images)} adversarial)")

        return train_images, train_labels

    def sample_adversarial_images(self, num_samples):
        """
        从优化的对抗分布采样视角并渲染图像
        使用distribution sharing策略
        """
        images = []
        labels = []

        samples_per_scene = num_samples // len(self.renderers)

        for scene_id, renderer in self.renderers.items():
            dist = self.adversarial_distributions[scene_id]
            gmvfool = self.gmvfools[scene_id]

            for _ in range(samples_per_scene):
                # Distribution sharing: 以概率π选择其他场景的分布
                if np.random.rand() < self.config['distribution_sharing_prob']:
                    # 随机选择同类场景的分布（如果有场景分类的话）
                    shared_scene = np.random.choice(list(self.adversarial_distributions.keys()))
                    dist = self.adversarial_distributions[shared_scene]

                # 从分布采样视角
                viewpoint = self.sample_from_distribution(dist)

                # 渲染图像
                rgb, semantic = renderer.render(viewpoint, return_semantic=True)

                # 从语义图生成YOLO标签
                yolo_label = self.semantic_to_yolo_labels(semantic)

                images.append(rgb)
                labels.append(yolo_label)

        return images, labels

    def sample_from_distribution(self, distribution):
        """
        从混合高斯分布采样视角参数
        """
        # 采样高斯分量
        k = np.random.choice(self.config['num_gaussian_components'],
                            p=distribution['omegas'].numpy())

        # 采样噪声
        mu_k = distribution['mus'][k].numpy()
        sigma_k = distribution['sigmas'][k].numpy()
        noise = np.random.randn(6)

        # 重参数化
        v = mu_k + sigma_k * noise
        v = np.tanh(v)  # 归一化

        # 映射到实际视角范围
        v_real = self.map_to_viewpoint_range(v)

        # 转换为相机pose矩阵
        camera_pose = self.viewpoint_to_camera_pose(v_real)

        return camera_pose

    def train_yolo_one_epoch(self, images, labels, epoch):
        """
        在一个epoch的数据上训练YOLO
        """
        # 保存临时训练数据
        temp_data_dir = Path('./tmp_train_data') / f'epoch_{epoch}'
        temp_data_dir.mkdir(parents=True, exist_ok=True)

        # 保存图像和标签（YOLO格式）
        self.save_yolo_format_data(images, labels, temp_data_dir)

        # 使用YOLO的训练接口
        train_results = self.yolo.train(
            data=str(temp_data_dir / 'data.yaml'),
            epochs=1,  # 只训练1个epoch
            imgsz=self.config['image_size'][0],
            batch=self.config['batch_size'],
            resume=True if epoch > 0 else False,
            device=self.device,
            verbose=True
        )

        print(f"Epoch {epoch+1} training completed. Loss: {train_results}")

    def evaluate(self, epoch):
        """
        评估模型性能
        在自然视角和对抗视角上测试
        """
        print(f"\n{'='*50}")
        print(f"Evaluation at Epoch {epoch+1}")
        print(f"{'='*50}")

        # 1. 在自然视角上评估
        natural_results = self.evaluate_on_natural_viewpoints()
        print(f"Natural Viewpoints - mAP: {natural_results['mAP']:.4f}")

        # 2. 在对抗视角上评估
        adversarial_results = self.evaluate_on_adversarial_viewpoints()
        print(f"Adversarial Viewpoints - mAP: {adversarial_results['mAP']:.4f}")

        # 3. 记录结果
        self.log_results(epoch, natural_results, adversarial_results)

    def evaluate_on_adversarial_viewpoints(self):
        """
        在对抗视角上评估YOLO性能
        """
        total_detections = 0
        correct_detections = 0
        all_precisions = []
        all_recalls = []

        for scene_id, gmvfool in self.gmvfools.items():
            # 采样对抗视角
            num_test_samples = 20
            dist = self.adversarial_distributions[scene_id]

            for _ in range(num_test_samples):
                viewpoint = self.sample_from_distribution(dist)
                rgb, semantic = self.renderers[scene_id].render(viewpoint)

                # YOLO推理
                results = self.yolo(rgb, verbose=False)[0]

                # 计算指标（与ground truth对比）
                gt_boxes = self.semantic_to_yolo_labels(semantic)
                pred_boxes = results.boxes

                # 计算mAP等指标
                precision, recall = self.compute_detection_metrics(pred_boxes, gt_boxes)
                all_precisions.append(precision)
                all_recalls.append(recall)

        # 计算平均性能
        mean_precision = np.mean(all_precisions)
        mean_recall = np.mean(all_recalls)
        mAP = (mean_precision + mean_recall) / 2  # 简化版mAP

        return {
            'mAP': mAP,
            'precision': mean_precision,
            'recall': mean_recall
        }

    # ============== 辅助函数 ==============

    def get_scene_target_objects(self, scene_id):
        """获取场景中的目标物体类别"""
        # TODO: 从GS3LAM的语义图中提取
        # 临时返回常见类别
        return [0, 56, 57, 58, 59, 60]  # COCO: person, chair, couch, etc.

    def semantic_to_yolo_labels(self, semantic_map):
        """
        将GS3LAM的语义图转换为YOLO格式标签
        返回: List of [class_id, x_center, y_center, width, height]
        """
        # TODO: 实现语义图到bounding box的转换
        # 可以使用连通域分析
        labels = []

        unique_classes = np.unique(semantic_map)
        for cls_id in unique_classes:
            if cls_id == 0:  # 背景类
                continue

            # 找到该类别的所有像素
            mask = (semantic_map == cls_id).astype(np.uint8)

            # 找bounding box
            import cv2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # 转换为YOLO格式（归一化）
                H, W = semantic_map.shape
                x_center = (x + w/2) / W
                y_center = (y + h/2) / H
                w_norm = w / W
                h_norm = h / H

                labels.append([cls_id, x_center, y_center, w_norm, h_norm])

        return labels

    def visualize_adversarial_views(self, scene_id, gmvfool, num_samples=10):
        """可视化对抗视角"""
        save_dir = Path(self.config['output_dir']) / 'adversarial_views' / scene_id
        save_dir.mkdir(parents=True, exist_ok=True)

        dist = self.adversarial_distributions[scene_id]

        for i in range(num_samples):
            viewpoint = self.sample_from_distribution(dist)
            rgb, _ = self.renderers[scene_id].render(viewpoint)

            # YOLO推理并可视化
            results = self.yolo(rgb, verbose=False)[0]
            annotated_img = results.plot()

            # 保存
            import cv2
            cv2.imwrite(str(save_dir / f'view_{i:03d}.jpg'), annotated_img)

    def save_model(self):
        """保存训练好的模型"""
        save_path = Path(self.config['output_dir']) / 'viat_yolo_final.pt'
        self.yolo.save(str(save_path))
        print(f"✅ Model saved to {save_path}")

    # 其他辅助函数（省略详细实现）
    def load_natural_images(self, num_samples):
        """加载自然视角图像"""
        # TODO: 实现
        return [], []

    def evaluate_on_natural_viewpoints(self):
        """在自然视角评估"""
        # TODO: 实现
        return {'mAP': 0.0}

    def compute_detection_metrics(self, pred_boxes, gt_boxes):
        """计算检测指标"""
        # TODO: 实现IoU计算和mAP
        return 0.0, 0.0

    def save_yolo_format_data(self, images, labels, save_dir):
        """保存YOLO格式数据"""
        # TODO: 实现
        pass

    def map_to_viewpoint_range(self, v):
        """映射视角参数"""
        # TODO: 实现
        return v

    def viewpoint_to_camera_pose(self, v):
        """视角参数转相机pose"""
        # TODO: 实现
        pose = np.eye(4)
        return pose

    def log_results(self, epoch, natural_results, adversarial_results):
        """记录训练结果"""
        # TODO: 使用tensorboard或wandb
        pass


# ============== 主函数 ==============

def main():
    """训练VIAT-YOLO"""
    # 加载配置
    config = {
        # YOLO配置
        'yolo_model': 'yolov8x.pt',  # 或 'yolov9c.pt', 'yolov8x-world.pt'
        'finetune_on_dataset': True,
        'yolo_data_yaml': './data/replica_yolo.yaml',
        'finetune_epochs': 50,

        # GS3LAM配置
        'scene_data_dir': './data/Replica',
        'gs3lam_checkpoint': None,  # 或指定checkpoint路径
        'image_size': (680, 1200),

        # GMVFool配置
        'num_gaussian_components': 15,
        'lambda_entropy': 0.01,
        'num_samples': 100,

        # 训练配置
        'num_epochs': 60,
        'phase1_iterations': 50,
        'phase2_iterations': 10,
        'scenes_per_epoch': 5,
        'learning_rate': 0.01,
        'batch_size': 16,

        # 数据配置
        'natural_samples_per_epoch': 512,
        'adversarial_samples_per_epoch': 512,
        'distribution_sharing_prob': 0.5,

        # 其他
        'eval_frequency': 5,
        'output_dir': './outputs/viat_yolo',
    }

    # 创建训练器
    trainer = VIATYOLOTrainer(config)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
