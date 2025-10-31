"""
GMVFool for Scene-level YOLO Detection with GS3LAM
使用YOLO检测置信度作为优化目标
"""

import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import cv2

class GMVFoolYOLO:
    """
    将VIAT的GMVFool扩展到场景级YOLO检测
    核心思路：优化视角参数，最大化YOLO检测的置信度下降
    """

    def __init__(self,
                 yolo_model_path='yolov8x.pt',  # 预训练YOLO模型
                 gs3lam_renderer=None,           # GS3LAM渲染器
                 num_components=15,              # 高斯混合分量数
                 lambda_entropy=0.01):           # 熵正则化系数

        # 1. 加载YOLO模型
        self.yolo = YOLO(yolo_model_path)
        self.yolo.to('cuda')

        # 2. GS3LAM渲染器
        self.renderer = gs3lam_renderer

        # 3. GMVFool参数
        self.K = num_components
        self.lambda_entropy = lambda_entropy

        # 4. 初始化高斯混合分布参数
        self.init_distribution()

    def init_distribution(self):
        """初始化混合高斯分布参数"""
        # 视角参数：[yaw, pitch, roll, x, y, z]
        self.omegas = torch.ones(self.K) / self.K  # 权重
        self.mus = torch.randn(self.K, 6) * 0.1    # 均值
        self.sigmas = torch.ones(self.K, 6) * 0.5  # 标准差

    def compute_detection_loss(self, image, target_objects):
        """
        计算YOLO检测loss - 核心创新点

        Args:
            image: 渲染的RGB图像 [H, W, 3]
            target_objects: 目标物体类别列表

        Returns:
            loss: 标量loss值（检测置信度的负数，越大表示检测越差）
        """
        # 1. YOLO推理
        results = self.yolo(image, verbose=False)[0]

        # 2. 提取检测结果
        boxes = results.boxes

        if len(boxes) == 0:
            # 没有检测到任何物体 - 攻击成功
            return 1.0  # 最大loss

        # 3. 计算目标物体的平均置信度
        target_confidences = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 只关注目标物体类别
            if cls_id in target_objects:
                target_confidences.append(conf)

        if len(target_confidences) == 0:
            return 1.0  # 目标物体未检测到

        # 4. Loss = 1 - 平均置信度（置信度越低，loss越高）
        avg_conf = np.mean(target_confidences)
        loss = 1.0 - avg_conf

        return loss

    def render_from_viewpoint(self, viewpoint_params):
        """
        从给定视角渲染图像（调用GS3LAM）

        Args:
            viewpoint_params: [yaw, pitch, roll, x, y, z]

        Returns:
            rgb_image: [H, W, 3]
            semantic_map: [H, W] 语义图（可选用于可视化）
        """
        # 将视角参数转换为相机pose
        camera_pose = self.viewpoint_to_camera_pose(viewpoint_params)

        # 调用GS3LAM渲染
        rgb_image, semantic_map = self.renderer.render(camera_pose)

        return rgb_image, semantic_map

    def viewpoint_to_camera_pose(self, viewpoint_params):
        """
        将视角参数转换为相机pose矩阵
        viewpoint_params: [yaw, pitch, roll, x, y, z]
        返回: 4x4变换矩阵
        """
        yaw, pitch, roll, x, y, z = viewpoint_params

        # 构建旋转矩阵（欧拉角转旋转矩阵）
        # Yaw (Z), Pitch (Y), Roll (X)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        # ZYX欧拉角旋转矩阵
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

        # 构建4x4齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T

    def optimize_adversarial_viewpoints(self,
                                       scene_id,
                                       target_objects,
                                       num_iterations=50,
                                       learning_rate=0.01,
                                       num_samples=100):
        """
        优化对抗视角分布（NES优化）

        Args:
            scene_id: 场景ID
            target_objects: 目标物体类别ID列表
            num_iterations: 优化迭代次数
            learning_rate: 学习率
            num_samples: 蒙特卡洛采样数

        Returns:
            optimized_distribution: 优化后的分布参数
        """
        print(f"Optimizing adversarial viewpoints for scene {scene_id}...")

        for iter in range(num_iterations):
            # 1. 采样视角参数
            gradients_omega = np.zeros(self.K)
            gradients_mu = np.zeros((self.K, 6))
            gradients_sigma = np.zeros((self.K, 6))

            for _ in range(num_samples):
                # 采样高斯分量
                k = np.random.choice(self.K, p=self.omegas.numpy())

                # 采样噪声
                r = np.random.randn(6)

                # 重参数化采样
                v = self.mus[k].numpy() + self.sigmas[k].numpy() * r
                v = np.tanh(v)  # 归一化到[-1, 1]

                # 映射到实际视角范围
                v_real = self.map_to_viewpoint_range(v)

                # 2. 渲染图像
                rgb_image, _ = self.render_from_viewpoint(v_real)

                # 3. 计算检测loss
                loss_cls = self.compute_detection_loss(rgb_image, target_objects)

                # 4. 计算熵正则项
                log_prob = self.compute_log_prob(v, k)
                loss_entropy = -self.lambda_entropy * log_prob

                # 总loss
                loss = loss_cls + loss_entropy

                # 5. 计算自然梯度（NES）
                gradients_omega[k] += loss / self.omegas[k]
                gradients_mu[k] += loss * self.sigmas[k].numpy() * r / self.omegas[k]
                gradients_sigma[k] += loss * self.sigmas[k].numpy() * (r**2 - 1) / (2 * self.omegas[k])

            # 6. 更新参数
            self.omegas += learning_rate * torch.tensor(gradients_omega / num_samples)
            self.mus += learning_rate * torch.tensor(gradients_mu / num_samples)
            self.sigmas += learning_rate * torch.tensor(gradients_sigma / num_samples)

            # 归一化权重
            self.omegas = torch.clamp(self.omegas, min=0)
            self.omegas /= self.omegas.sum()

            # 打印进度
            if (iter + 1) % 10 == 0:
                print(f"Iteration {iter+1}/{num_iterations}, Avg Loss: {loss:.4f}")

        return {
            'omegas': self.omegas,
            'mus': self.mus,
            'sigmas': self.sigmas
        }

    def map_to_viewpoint_range(self, v_normalized):
        """
        将归一化的视角参数映射到实际范围
        v_normalized: [-1, 1]^6
        返回: 实际视角参数
        """
        # 定义视角范围
        v_min = np.array([-np.pi, -np.pi/6, -np.pi/6, -2, -2, -2])  # [yaw, pitch, roll, x, y, z]
        v_max = np.array([np.pi, np.pi/6, np.pi/6, 2, 2, 2])

        # 线性映射
        v_real = (v_normalized + 1) / 2 * (v_max - v_min) + v_min

        return v_real

    def compute_log_prob(self, v, k):
        """计算对数概率（用于熵正则）"""
        mu_k = self.mus[k].numpy()
        sigma_k = self.sigmas[k].numpy()

        log_prob = -0.5 * np.sum((v - mu_k)**2 / sigma_k**2)
        log_prob += -0.5 * 6 * np.log(2 * np.pi) - np.sum(np.log(sigma_k))

        return log_prob


# ============== 使用示例 ==============

def example_usage():
    """示例：如何使用GMVFoolYOLO"""

    # 1. 初始化GS3LAM渲染器（需要你实现GS3LAM的接口）
    from gs3lam_interface import GS3LAMRenderer
    gs3lam = GS3LAMRenderer(scene_path='./data/Replica/office0')

    # 2. 初始化GMVFoolYOLO
    gmvfool = GMVFoolYOLO(
        yolo_model_path='yolov8x.pt',  # 预训练YOLO
        gs3lam_renderer=gs3lam,
        num_components=15,
        lambda_entropy=0.01
    )

    # 3. 定义目标物体（例如：person, chair, laptop在COCO中的ID）
    target_objects = [0, 56, 63]  # COCO类别ID

    # 4. 优化对抗视角分布
    optimized_dist = gmvfool.optimize_adversarial_viewpoints(
        scene_id='office0',
        target_objects=target_objects,
        num_iterations=50,
        learning_rate=0.01,
        num_samples=100
    )

    print("Optimization completed!")
    print(f"Optimized distribution weights: {optimized_dist['omegas']}")


if __name__ == '__main__':
    example_usage()
