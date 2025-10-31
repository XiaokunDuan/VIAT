"""
从GS3LAM输出准备YOLO格式数据集
将语义分割图转换为目标检测的bounding box格式
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import json


class SemanticToYOLOConverter:
    """语义图到YOLO格式转换器"""

    def __init__(self,
                 class_mapping=None,
                 min_box_area=100,
                 max_box_aspect_ratio=10.0):
        """
        Args:
            class_mapping: {semantic_class_id: yolo_class_id} 映射
            min_box_area: 最小box面积（像素），过滤太小的检测
            max_box_aspect_ratio: 最大长宽比，过滤异常box
        """
        self.class_mapping = class_mapping or self.get_default_replica_mapping()
        self.min_box_area = min_box_area
        self.max_box_aspect_ratio = max_box_aspect_ratio

    @staticmethod
    def get_default_replica_mapping():
        """
        Replica数据集的默认类别映射
        参考: https://github.com/facebookresearch/Replica-Dataset
        """
        # Replica语义类别 -> YOLO类别ID
        # 这里列出常见的物体类别
        return {
            3: 0,    # chair -> class 0
            5: 1,    # table -> class 1
            10: 2,   # sofa -> class 2
            11: 3,   # bed -> class 3
            14: 4,   # cabinet -> class 4
            18: 5,   # tv -> class 5
            23: 6,   # plant -> class 6
            26: 7,   # lamp -> class 7
            31: 8,   # desk -> class 8
            34: 9,   # shelf -> class 9
            # 可以根据实际需要添加更多类别
        }

    def semantic_to_boxes(self, semantic_map):
        """
        将语义分割图转换为YOLO bounding box格式

        Args:
            semantic_map: [H, W] numpy array of class IDs

        Returns:
            boxes: List of [class_id, x_center, y_center, width, height]
                   所有值归一化到[0, 1]
        """
        H, W = semantic_map.shape
        boxes = []

        unique_classes = np.unique(semantic_map)

        for sem_class in unique_classes:
            # 跳过背景类(0)或未映射的类别
            if sem_class == 0 or sem_class not in self.class_mapping:
                continue

            # 创建该类别的二值mask
            mask = (semantic_map == sem_class).astype(np.uint8) * 255

            # 形态学操作（去噪）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 查找连通域
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # 计算面积
                area = cv2.contourArea(contour)
                if area < self.min_box_area:
                    continue

                # 获取外接矩形
                x, y, w, h = cv2.boundingRect(contour)

                # 检查长宽比
                aspect_ratio = max(w/h, h/w) if h > 0 else 0
                if aspect_ratio > self.max_box_aspect_ratio:
                    continue

                # 转换为YOLO格式（归一化）
                x_center = (x + w/2) / W
                y_center = (y + h/2) / H
                w_norm = w / W
                h_norm = h / H

                # YOLO类别ID
                yolo_class = self.class_mapping[sem_class]

                boxes.append([yolo_class, x_center, y_center, w_norm, h_norm])

        return boxes

    def visualize_boxes(self, image, boxes):
        """
        可视化检测框（用于调试）

        Args:
            image: RGB图像 [H, W, 3]
            boxes: YOLO格式的boxes

        Returns:
            vis_image: 绘制了boxes的图像
        """
        vis_img = image.copy()
        H, W = vis_img.shape[:2]

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        for box in boxes:
            cls_id, x_c, y_c, w, h = box

            # 反归一化
            x_c *= W
            y_c *= H
            w *= W
            h *= H

            # 计算左上角和右下角
            x1 = int(x_c - w/2)
            y1 = int(y_c - h/2)
            x2 = int(x_c + w/2)
            y2 = int(y_c + h/2)

            # 绘制矩形
            color = colors[int(cls_id) % len(colors)]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # 标注类别
            label = f"Class {int(cls_id)}"
            cv2.putText(vis_img, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_img


class YOLODatasetBuilder:
    """YOLO数据集构建器"""

    def __init__(self,
                 gs3lam_output_dir,
                 save_dir,
                 train_split=0.8,
                 class_names=None):
        """
        Args:
            gs3lam_output_dir: GS3LAM输出目录
            save_dir: YOLO数据集保存目录
            train_split: 训练集比例
            class_names: 类别名称列表
        """
        self.gs3lam_dir = Path(gs3lam_output_dir)
        self.save_dir = Path(save_dir)
        self.train_split = train_split

        # 默认Replica类别名称
        self.class_names = class_names or [
            'chair', 'table', 'sofa', 'bed', 'cabinet',
            'tv', 'plant', 'lamp', 'desk', 'shelf'
        ]

        # 初始化转换器
        self.converter = SemanticToYOLOConverter()

        # 创建目录结构
        self._create_directory_structure()

    def _create_directory_structure(self):
        """创建YOLO数据集目录结构"""
        for split in ['train', 'val']:
            (self.save_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.save_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

        # 可视化目录（用于调试）
        (self.save_dir / 'visualizations').mkdir(parents=True, exist_ok=True)

    def build_dataset(self, visualize_samples=20):
        """
        构建完整的YOLO数据集

        Args:
            visualize_samples: 可视化的样本数（用于检查）
        """
        print("Building YOLO dataset from GS3LAM outputs...")
        print(f"Source: {self.gs3lam_dir}")
        print(f"Target: {self.save_dir}")

        # 收集所有场景
        scene_dirs = [d for d in self.gs3lam_dir.iterdir() if d.is_dir()]
        print(f"Found {len(scene_dirs)} scenes")

        # 统计信息
        stats = {
            'total_images': 0,
            'total_boxes': 0,
            'train_images': 0,
            'val_images': 0,
            'class_distribution': {}
        }

        # 处理每个场景
        all_samples = []
        for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
            samples = self._process_scene(scene_dir)
            all_samples.extend(samples)

        # 划分训练集和验证集
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * self.train_split)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

        # 保存数据
        stats['train_images'] = self._save_samples(train_samples, 'train', stats)
        stats['val_images'] = self._save_samples(val_samples, 'val', stats)
        stats['total_images'] = len(all_samples)

        # 创建data.yaml
        self._create_data_yaml()

        # 可视化一些样本
        self._visualize_samples(all_samples[:visualize_samples])

        # 保存统计信息
        self._save_stats(stats)

        print("\n" + "="*50)
        print("✅ Dataset building completed!")
        print(f"Total images: {stats['total_images']}")
        print(f"  Train: {stats['train_images']}")
        print(f"  Val: {stats['val_images']}")
        print(f"Total boxes: {stats['total_boxes']}")
        print(f"Saved to: {self.save_dir}")
        print("="*50)

    def _process_scene(self, scene_dir):
        """
        处理单个场景

        Returns:
            samples: List of (rgb_path, semantic_path, scene_name)
        """
        samples = []

        # 查找RGB和语义图
        rgb_dir = scene_dir / 'rgb'
        semantic_dir = scene_dir / 'semantic'

        if not rgb_dir.exists() or not semantic_dir.exists():
            print(f"⚠️ Warning: {scene_dir.name} missing rgb or semantic dir")
            return samples

        # 匹配RGB和语义图文件
        rgb_files = sorted(rgb_dir.glob('*.png'))
        semantic_files = sorted(semantic_dir.glob('*.npy'))

        # 确保配对
        rgb_dict = {f.stem: f for f in rgb_files}
        sem_dict = {f.stem: f for f in semantic_files}

        for stem in rgb_dict.keys():
            if stem in sem_dict:
                samples.append({
                    'rgb_path': rgb_dict[stem],
                    'semantic_path': sem_dict[stem],
                    'scene_name': scene_dir.name,
                    'frame_id': stem
                })

        return samples

    def _save_samples(self, samples, split, stats):
        """保存样本到指定split"""
        count = 0

        for sample in tqdm(samples, desc=f"Saving {split} set"):
            # 读取RGB图像
            rgb = cv2.imread(str(sample['rgb_path']))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # 读取语义图
            semantic = np.load(str(sample['semantic_path']))

            # 转换为YOLO boxes
            boxes = self.converter.semantic_to_boxes(semantic)

            if len(boxes) == 0:
                continue  # 跳过没有目标的图像

            # 生成文件名
            filename = f"{sample['scene_name']}_{sample['frame_id']}"

            # 保存图像
            img_save_path = self.save_dir / 'images' / split / f'{filename}.jpg'
            cv2.imwrite(str(img_save_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # 保存标签
            label_save_path = self.save_dir / 'labels' / split / f'{filename}.txt'
            with open(label_save_path, 'w') as f:
                for box in boxes:
                    f.write(' '.join(map(str, box)) + '\n')

            # 更新统计
            stats['total_boxes'] += len(boxes)
            for box in boxes:
                cls_id = int(box[0])
                stats['class_distribution'][cls_id] = \
                    stats['class_distribution'].get(cls_id, 0) + 1

            count += 1

        return count

    def _create_data_yaml(self):
        """创建YOLO数据配置文件"""
        data_yaml = {
            'path': str(self.save_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }

        yaml_path = self.save_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"✅ Created data.yaml at {yaml_path}")

    def _visualize_samples(self, samples):
        """可视化样本（用于检查数据质量）"""
        print(f"Visualizing {len(samples)} samples...")

        for i, sample in enumerate(samples):
            # 读取数据
            rgb = cv2.imread(str(sample['rgb_path']))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            semantic = np.load(str(sample['semantic_path']))

            # 转换
            boxes = self.converter.semantic_to_boxes(semantic)

            # 可视化
            vis_img = self.converter.visualize_boxes(rgb, boxes)

            # 保存
            save_path = self.save_dir / 'visualizations' / f'sample_{i:03d}.jpg'
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        print(f"✅ Visualizations saved to {self.save_dir / 'visualizations'}")

    def _save_stats(self, stats):
        """保存数据集统计信息"""
        stats_path = self.save_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✅ Stats saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare YOLO dataset from GS3LAM outputs'
    )
    parser.add_argument(
        '--gs3lam_dir',
        type=str,
        required=True,
        help='GS3LAM output directory'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='YOLO dataset save directory'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Train set split ratio'
    )
    parser.add_argument(
        '--visualize',
        type=int,
        default=20,
        help='Number of samples to visualize'
    )

    args = parser.parse_args()

    # 构建数据集
    builder = YOLODatasetBuilder(
        gs3lam_output_dir=args.gs3lam_dir,
        save_dir=args.save_dir,
        train_split=args.train_split
    )

    builder.build_dataset(visualize_samples=args.visualize)


if __name__ == '__main__':
    main()
