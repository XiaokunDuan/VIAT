"""
GS3LAM接口适配器
将GS3LAM的输出适配为GMVFoolYOLO可用的格式
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../GS3LAM'))

import torch
import numpy as np
import cv2
from pathlib import Path


class GS3LAMRenderer:
    """
    GS3LAM渲染器包装类
    提供统一的渲染接口给GMVFoolYOLO使用
    """

    def __init__(self,
                 scene_path,
                 config_path=None,
                 checkpoint_path=None,
                 image_size=(680, 1200)):  # Replica默认分辨率
        """
        Args:
            scene_path: 场景数据路径（如 './data/Replica/office0'）
            config_path: GS3LAM配置文件路径
            checkpoint_path: 预训练的GS3LAM checkpoint路径
            image_size: 输出图像大小 (H, W)
        """
        self.scene_path = Path(scene_path)
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载GS3LAM模型
        self._load_gs3lam_model(config_path, checkpoint_path)

        print(f"✅ GS3LAM renderer initialized for scene: {scene_path}")

    def _load_gs3lam_model(self, config_path, checkpoint_path):
        """
        加载GS3LAM模型（需要根据实际GS3LAM代码调整）
        """
        try:
            # 方案1: 如果GS3LAM已经训练好，加载checkpoint
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading GS3LAM from checkpoint: {checkpoint_path}")
                # TODO: 根据GS3LAM的实际API加载
                # self.gs3lam_model = load_gs3lam_checkpoint(checkpoint_path)
                pass

            # 方案2: 在线运行GS3LAM（如果需要实时SLAM）
            else:
                print("Initializing GS3LAM model...")
                # TODO: 根据GS3LAM的实际API初始化
                # from src.slam import GaussianSLAM
                # self.gs3lam_model = GaussianSLAM(config_path)
                pass

        except Exception as e:
            print(f"⚠️ Warning: Failed to load GS3LAM model: {e}")
            print("Using fallback rendering (will return dummy images)")
            self.use_fallback = True

    def render(self, camera_pose, return_semantic=True):
        """
        从指定相机位姿渲染图像

        Args:
            camera_pose: 4x4相机变换矩阵 (numpy array)
            return_semantic: 是否返回语义图

        Returns:
            rgb_image: RGB图像 [H, W, 3], numpy array, uint8
            semantic_map: 语义图 [H, W], numpy array, int (如果return_semantic=True)
        """
        # 转换为torch tensor
        pose_tensor = torch.from_numpy(camera_pose).float().to(self.device)

        # 调用GS3LAM渲染
        # TODO: 根据GS3LAM实际API调整
        # rendered = self.gs3lam_model.render(pose_tensor)
        # rgb_image = rendered['rgb'].cpu().numpy()
        # semantic_map = rendered['semantic'].cpu().numpy()

        # Fallback: 返回模拟数据（用于测试）
        rgb_image = self._render_fallback(camera_pose)
        semantic_map = self._render_semantic_fallback(camera_pose)

        if return_semantic:
            return rgb_image, semantic_map
        else:
            return rgb_image, None

    def _render_fallback(self, camera_pose):
        """
        Fallback渲染函数（用于测试，返回模拟图像）
        实际使用时应该被真实的GS3LAM渲染替换
        """
        # 创建一个带网格的测试图像
        img = np.ones((*self.image_size, 3), dtype=np.uint8) * 200

        # 绘制网格
        grid_size = 50
        for i in range(0, self.image_size[0], grid_size):
            cv2.line(img, (0, i), (self.image_size[1], i), (150, 150, 150), 1)
        for j in range(0, self.image_size[1], grid_size):
            cv2.line(img, (j, 0), (j, self.image_size[0]), (150, 150, 150), 1)

        # 在图像上显示相机位姿信息
        pose_text = f"Pose: [{camera_pose[0,3]:.2f}, {camera_pose[1,3]:.2f}, {camera_pose[2,3]:.2f}]"
        cv2.putText(img, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加一些随机的"物体"区域（用于测试YOLO）
        num_objects = np.random.randint(3, 8)
        for _ in range(num_objects):
            x1 = np.random.randint(0, self.image_size[1] - 200)
            y1 = np.random.randint(0, self.image_size[0] - 200)
            x2 = x1 + np.random.randint(100, 200)
            y2 = y1 + np.random.randint(100, 200)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        return img

    def _render_semantic_fallback(self, camera_pose):
        """
        Fallback语义图渲染（用于测试）
        """
        # 创建随机语义图（模拟不同的物体类别）
        semantic = np.random.randint(0, 20, self.image_size, dtype=np.int32)
        return semantic

    def batch_render(self, camera_poses, return_semantic=True):
        """
        批量渲染多个视角

        Args:
            camera_poses: List of 4x4 camera poses
            return_semantic: 是否返回语义图

        Returns:
            rgb_images: List of RGB images
            semantic_maps: List of semantic maps (if return_semantic=True)
        """
        rgb_images = []
        semantic_maps = []

        for pose in camera_poses:
            rgb, sem = self.render(pose, return_semantic)
            rgb_images.append(rgb)
            if return_semantic:
                semantic_maps.append(sem)

        return rgb_images, semantic_maps if return_semantic else None

    def get_initial_viewpoint(self):
        """
        获取场景的初始/自然视角

        Returns:
            camera_pose: 4x4变换矩阵
        """
        # 默认初始视角（可根据场景调整）
        pose = np.eye(4)
        pose[2, 3] = 3.0  # z方向偏移3米
        return pose

    def export_trajectory_images(self,
                                  viewpoint_distribution,
                                  num_samples=100,
                                  save_dir='./adversarial_views'):
        """
        从优化后的对抗视角分布采样并导出图像

        Args:
            viewpoint_distribution: GMVFoolYOLO优化的分布参数
            num_samples: 采样数量
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting {num_samples} adversarial viewpoint images...")

        for i in range(num_samples):
            # 从分布采样视角
            # TODO: 实现从混合高斯分布采样
            viewpoint = self._sample_from_distribution(viewpoint_distribution)

            # 渲染图像
            rgb, semantic = self.render(viewpoint, return_semantic=True)

            # 保存
            cv2.imwrite(str(save_dir / f'rgb_{i:04d}.png'),
                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(str(save_dir / f'semantic_{i:04d}.npy'), semantic)

        print(f"✅ Images saved to {save_dir}")

    def _sample_from_distribution(self, distribution):
        """从混合高斯分布采样视角参数"""
        # TODO: 实现采样逻辑
        # 临时返回随机pose
        pose = np.eye(4)
        pose[:3, 3] = np.random.randn(3) * 0.5
        return pose


# ============== GS3LAM快速集成工具 ==============

class GS3LAMQuickIntegration:
    """
    快速集成工具：直接使用GS3LAM的输出数据
    适用于GS3LAM已经运行完成的场景
    """

    def __init__(self, gs3lam_output_dir):
        """
        Args:
            gs3lam_output_dir: GS3LAM输出目录（包含重建结果）
        """
        self.output_dir = Path(gs3lam_output_dir)
        self.load_gs3lam_results()

    def load_gs3lam_results(self):
        """
        加载GS3LAM的输出结果
        假设GS3LAM保存了：
        - gaussian_field.ply: 3D高斯点云
        - poses.txt: 相机轨迹
        - semantic_labels.npy: 语义标签
        """
        print(f"Loading GS3LAM results from {self.output_dir}")

        # 加载高斯场景表示
        gaussian_path = self.output_dir / 'gaussian_field.ply'
        if gaussian_path.exists():
            # TODO: 加载高斯场景
            pass

        # 加载相机轨迹
        pose_path = self.output_dir / 'poses.txt'
        if pose_path.exists():
            self.poses = np.loadtxt(pose_path)
            print(f"Loaded {len(self.poses)} camera poses")

        # 加载渲染图像（如果已经预渲染）
        self.rgb_dir = self.output_dir / 'rgb'
        self.semantic_dir = self.output_dir / 'semantic'

    def render_from_pose_index(self, pose_idx):
        """
        从保存的pose索引渲染图像
        如果GS3LAM已经渲染好，直接读取
        """
        rgb_path = self.rgb_dir / f'{pose_idx:06d}.png'
        semantic_path = self.semantic_dir / f'{pose_idx:06d}.npy'

        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            rgb = None

        if semantic_path.exists():
            semantic = np.load(semantic_path)
        else:
            semantic = None

        return rgb, semantic


# ============== 测试代码 ==============

def test_gs3lam_renderer():
    """测试GS3LAM渲染器"""
    print("Testing GS3LAM Renderer...")

    # 初始化渲染器（使用fallback模式测试）
    renderer = GS3LAMRenderer(
        scene_path='./data/Replica/office0',
        image_size=(480, 640)
    )

    # 测试渲染
    initial_pose = renderer.get_initial_viewpoint()
    rgb, semantic = renderer.render(initial_pose)

    print(f"Rendered RGB shape: {rgb.shape}")
    print(f"Rendered Semantic shape: {semantic.shape}")

    # 保存测试图像
    cv2.imwrite('/tmp/test_render.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("✅ Test image saved to /tmp/test_render.png")

    # 测试批量渲染
    poses = [initial_pose for _ in range(5)]
    rgb_list, sem_list = renderer.batch_render(poses)
    print(f"✅ Batch rendered {len(rgb_list)} images")


if __name__ == '__main__':
    test_gs3lam_renderer()
