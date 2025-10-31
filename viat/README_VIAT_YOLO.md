# VIAT-YOLO: 场景级旋转鲁棒目标检测

## 📋 项目概述

本项目将VIAT框架从**物体级分类**扩展到**场景级目标检测**，研究大场景旋转下的YOLO鲁棒性。

### 核心创新点

1. **场景级扩展**：从单物体NeRF → GS3LAM场景重建
2. **任务扩展**：从图像分类 → 目标检测
3. **模型替换**：从ResNet/ViT → YOLO系列
4. **Loss设计**：从分类loss → 检测置信度loss

---

## 🎯 你的想法可行性分析

### ✅ 完全可行的部分

1. **GS3LAM输出图像和语义图** - GS3LAM天然支持这个功能
2. **场景级视角旋转** - 比物体级更有实际应用价值
3. **YOLO替换分类器** - YOLO更适合场景理解
4. **NES优化框架** - 可以适配YOLO的loss

### ⚠️ 需要解决的挑战

#### 挑战1: YOLO预训练模型适配问题

**问题**：YOLO预训练在COCO数据集（80类），但你的场景可能包含：
- 不同的类别集合
- 不同的类别分布
- 场景特定的物体

**解决方案（3种策略）**：

##### 策略A：直接使用COCO预训练（推荐用于快速原型）

```python
# 优点：最简单，无需微调
# 缺点：可能有些场景物体不在COCO中

from ultralytics import YOLO

# 直接使用预训练YOLO
model = YOLO('yolov8x.pt')  # 80类COCO

# 在推理时只关注场景中存在的类别
target_classes = [0, 56, 57, 58, 59, 60]  # person, chair, couch, bed, table, tv

# 在计算loss时只考虑这些类别的置信度
loss = compute_loss_for_target_classes(model, image, target_classes)
```

**适用场景**：
- Replica/ScanNet等室内场景（大部分物体在COCO中）
- 快速验证想法

##### 策略B：在场景数据上微调YOLO（推荐用于最佳性能）

```python
# 优点：适配特定场景，性能最好
# 缺点：需要标注数据

# 1. 准备场景数据集（YOLO格式）
# 从GS3LAM的语义图自动生成bounding box

def prepare_scene_dataset():
    """
    从GS3LAM语义图生成YOLO训练数据
    """
    for scene in scenes:
        # 加载语义图
        semantic_map = load_semantic(scene)

        # 转换为bounding box（连通域分析）
        boxes = semantic_to_boxes(semantic_map)

        # 保存为YOLO格式
        # 格式: class_id x_center y_center width height (归一化)
        save_yolo_labels(boxes, f'{scene}/labels')

# 2. 创建数据配置文件
# data/replica_yolo.yaml
"""
train: ./data/Replica/train/images
val: ./data/Replica/val/images

nc: 20  # 场景类别数
names: ['chair', 'table', 'sofa', 'bed', ...]
"""

# 3. 微调YOLO
model = YOLO('yolov8x.pt')
results = model.train(
    data='./data/replica_yolo.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)
```

**数据准备脚本**：

```python
# viat/prepare_yolo_dataset.py

import numpy as np
import cv2
from pathlib import Path

def semantic_map_to_yolo_boxes(semantic_map, class_mapping):
    """
    将GS3LAM的语义图转换为YOLO bounding box格式

    Args:
        semantic_map: [H, W] 语义分割图
        class_mapping: {semantic_id: yolo_class_id}

    Returns:
        boxes: List of [class_id, x_center, y_center, width, height]
    """
    H, W = semantic_map.shape
    boxes = []

    unique_classes = np.unique(semantic_map)

    for sem_id in unique_classes:
        if sem_id == 0 or sem_id not in class_mapping:
            continue  # 跳过背景或未映射类别

        # 提取该类别的mask
        mask = (semantic_map == sem_id).astype(np.uint8)

        # 找连通域（一个类别可能有多个实例）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 过滤太小的区域
            area = cv2.contourArea(contour)
            if area < 100:  # 像素数阈值
                continue

            # 计算bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # 转换为YOLO格式（归一化）
            x_center = (x + w/2) / W
            y_center = (y + h/2) / H
            w_norm = w / W
            h_norm = h / H

            yolo_class = class_mapping[sem_id]
            boxes.append([yolo_class, x_center, y_center, w_norm, h_norm])

    return boxes

def prepare_replica_yolo_dataset(gs3lam_output_dir, save_dir):
    """
    从GS3LAM输出准备YOLO数据集
    """
    save_dir = Path(save_dir)
    (save_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (save_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (save_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (save_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # Replica语义类别 -> YOLO类别映射
    class_mapping = {
        1: 0,   # chair
        2: 1,   # table
        3: 2,   # sofa
        # ... 根据实际场景定义
    }

    # 处理所有场景
    scenes = list(Path(gs3lam_output_dir).glob('*'))
    split_idx = int(len(scenes) * 0.8)  # 80% train, 20% val

    for idx, scene_dir in enumerate(scenes):
        split = 'train' if idx < split_idx else 'val'

        # 读取GS3LAM渲染的RGB和语义图
        rgb_files = sorted((scene_dir / 'rgb').glob('*.png'))
        semantic_files = sorted((scene_dir / 'semantic').glob('*.npy'))

        for rgb_path, sem_path in zip(rgb_files, semantic_files):
            # 复制RGB图像
            img_save_path = save_dir / 'images' / split / f'{scene_dir.name}_{rgb_path.name}'
            import shutil
            shutil.copy(rgb_path, img_save_path)

            # 生成YOLO标签
            semantic = np.load(sem_path)
            boxes = semantic_map_to_yolo_boxes(semantic, class_mapping)

            # 保存标签
            label_save_path = save_dir / 'labels' / split / f'{scene_dir.name}_{rgb_path.stem}.txt'
            with open(label_save_path, 'w') as f:
                for box in boxes:
                    f.write(' '.join(map(str, box)) + '\n')

    # 创建data.yaml
    yaml_content = f"""
train: {save_dir}/images/train
val: {save_dir}/images/val

nc: {len(class_mapping)}
names: {list(class_mapping.values())}
"""

    with open(save_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"✅ YOLO dataset prepared at {save_dir}")

# 使用方法
if __name__ == '__main__':
    prepare_replica_yolo_dataset(
        gs3lam_output_dir='./GS3LAM/outputs',
        save_dir='./data/Replica_YOLO'
    )
```

##### 策略C：使用开放词汇检测模型（最前沿）

```python
# 优点：无需标注，支持任意类别
# 缺点：性能可能略低，依赖文本描述

from ultralytics import YOLO

# 使用YOLO-World（支持开放词汇）
model = YOLO('yolov8x-worldv2.pt')

# 设置自定义类别（用自然语言描述）
model.set_classes([
    "office chair",
    "wooden table",
    "leather sofa",
    "computer monitor",
    "desk lamp"
])

# 直接检测，无需微调！
results = model(image)
```

---

## 🚀 完整实现流程

### 步骤1: 环境准备

```bash
# 1. 安装YOLO
pip install ultralytics

# 2. 安装GS3LAM依赖（参考GS3LAM/README.md）
cd GS3LAM
pip install -r requirements.txt
pip install submodules/gaussian-semantic-rasterization

# 3. 安装其他依赖
pip install tensorboard wandb opencv-python
```

### 步骤2: 准备场景数据

#### 选项A: 使用Replica数据集（推荐）

```bash
# 下载Replica数据集（带语义标签）
# https://huggingface.co/datasets/3David14/GS3LAM-Replica

# 运行GS3LAM重建
cd GS3LAM
python run.py configs/Replica/office0.py

# 导出渲染结果
python visualizer/offline_recon.py --mode sem_color --logdir ./outputs/office0
```

#### 选项B: 使用自己的数据

```bash
# 1. 准备RGB-D序列 + 语义标签
# 2. 运行GS3LAM
# 3. 导出多视角渲染
```

### 步骤3: 准备YOLO数据集

```bash
cd viat

# 从GS3LAM输出生成YOLO训练数据
python prepare_yolo_dataset.py \
    --gs3lam_dir ../GS3LAM/outputs \
    --save_dir ../data/Replica_YOLO
```

### 步骤4: 微调YOLO（如果使用策略B）

```bash
# 方法1: 使用Ultralytics CLI
yolo detect train \
    data=../data/Replica_YOLO/data.yaml \
    model=yolov8x.pt \
    epochs=50 \
    imgsz=640 \
    batch=16

# 方法2: 使用Python API
python -c "
from ultralytics import YOLO
model = YOLO('yolov8x.pt')
model.train(data='../data/Replica_YOLO/data.yaml', epochs=50)
"
```

### 步骤5: 运行VIAT-YOLO训练

```bash
# 编辑配置文件
vim config_viat_yolo.yaml

# 运行训练
python VIAT_YOLO_trainer.py --config config_viat_yolo.yaml

# 或者使用默认配置
python VIAT_YOLO_trainer.py
```

### 步骤6: 评估

```bash
# 评估在对抗视角上的性能
python evaluate_viat_yolo.py \
    --model ./outputs/viat_yolo/viat_yolo_final.pt \
    --test_scenes office0,office1,room0

# 生成对抗视角benchmark
python generate_adversarial_benchmark.py \
    --model ./outputs/viat_yolo/viat_yolo_final.pt \
    --num_samples 1000
```

---

## 📊 实验设计建议

### 对比实验

1. **Baseline**: 标准YOLO（COCO预训练）
2. **+Finetune**: 在场景数据微调的YOLO
3. **+Random Aug**: 随机视角数据增强
4. **+VIAT (Ours)**: 对抗视角训练

### 评估指标

```python
metrics = {
    # 检测性能
    'mAP': 平均精度,
    'mAP50': IoU=0.5的mAP,
    'mAP75': IoU=0.75的mAP,

    # 鲁棒性指标（新增）
    'detection_rate_drop': 检测率下降（自然 vs 对抗视角）,
    'confidence_drop': 置信度下降,
    'worst_case_mAP': 最差视角的mAP,

    # 视角分析
    'rotation_robustness_curve': 旋转角度 vs mAP曲线,
    'critical_viewpoints': 关键失败视角分布
}
```

### 可视化

```python
# 1. 对抗视角热图
# 显示哪些视角YOLO失败最严重

# 2. 检测置信度对比
# 自然视角 vs 对抗视角的置信度分布

# 3. 失败案例分析
# 可视化YOLO在对抗视角下的错检/漏检

# 4. 训练过程曲线
# Loss、mAP、鲁棒性指标随epoch变化
```

---

## 💡 进阶优化建议

### 1. 更强的对抗攻击

```python
# 除了GMVFool，还可以尝试：

# A. 联合视角+光照攻击
class GMVFoolLight(GMVFoolYOLO):
    def __init__(self):
        super().__init__()
        # 优化视角 + 光照参数
        self.light_params = ...

# B. 考虑遮挡
class GMVFoolOcclusion(GMVFoolYOLO):
    def optimize(self):
        # 找到被遮挡的关键视角
        ...
```

### 2. 多任务学习

```python
# 同时优化检测 + 分割
model = YOLOv8_SegmentationModel()

# Loss = Detection Loss + Segmentation Loss
loss = lambda_det * det_loss + lambda_seg * seg_loss
```

### 3. Transformer-based YOLO

```python
# 使用更强的YOLO变体
model = YOLO('yolov9e.pt')  # YOLOv9最大模型
model = YOLO('yolov10x.pt') # 最新版本
```

---

## 🐛 常见问题

### Q1: GS3LAM输出的语义图格式是什么？

**A**: GS3LAM输出的是整数类别ID的语义分割图，格式为`[H, W]`的numpy数组。需要转换为YOLO的bounding box格式。

### Q2: 如何处理类别不平衡？

**A**:
```python
# 使用类别权重
class_weights = compute_class_weights(semantic_maps)
loss = weighted_cross_entropy(pred, target, class_weights)

# 或使用Focal Loss
from ultralytics.utils.loss import FocalLoss
```

### Q3: GMVFool优化太慢怎么办？

**A**:
```python
# 1. 减少采样数
num_samples = 50  # 从100降到50

# 2. 减少高斯分量
num_components = 8  # 从15降到8

# 3. 使用更小的YOLO模型用于优化
attack_model = YOLO('yolov8n.pt')  # 用小模型找对抗视角
train_model = YOLO('yolov8x.pt')   # 用大模型训练
```

### Q4: 内存不足怎么办？

**A**:
```python
# 1. 降低图像分辨率
image_size = (480, 640)  # 而不是 (680, 1200)

# 2. 减小batch size
batch_size = 8

# 3. 使用梯度累积
gradient_accumulation_steps = 4

# 4. 使用混合精度训练
import torch
scaler = torch.cuda.amp.GradScaler()
```

---

## 📈 预期结果

基于VIAT原论文和YOLO的特性，预期：

1. **自然视角性能**: 略低于标准YOLO（~2-3% mAP）
2. **对抗视角性能**: 显著提升（~15-20% mAP）
3. **计算开销**: 训练时间增加2-3倍
4. **Transformer-YOLO**: 比CNN-YOLO更鲁棒

---

## 📚 参考资料

1. VIAT论文: https://arxiv.org/abs/2307.11528
2. GS3LAM论文: https://dl.acm.org/doi/10.1145/3664647.3680739
3. YOLO文档: https://docs.ultralytics.com/
4. Replica数据集: https://github.com/facebookresearch/Replica-Dataset

---

## 🤝 贡献

如有问题或改进建议，欢迎提Issue或PR！

---

## ✅ 快速检查清单

- [ ] 安装了YOLO和GS3LAM环境
- [ ] 下载了Replica数据集
- [ ] 运行了GS3LAM重建
- [ ] 准备了YOLO格式数据
- [ ] 微调了YOLO（可选）
- [ ] 测试了GMVFool攻击
- [ ] 开始VIAT训练
- [ ] 评估了对抗鲁棒性

祝研究顺利！🚀
