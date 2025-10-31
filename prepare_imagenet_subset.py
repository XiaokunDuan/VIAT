# /root/workspace/VIAT/prepare_imagenet_subset.py

import os

# 这是我们从 .sh 脚本中手动整理出的权威映射表
# 格式： '人类可读名': '项目数字ID'
NAME_TO_ID_MAP = {
    'airliner': '01', 'rifle': '03', 'barbell': '04', 'barrel': '06',
    'garden_cart': '07', 'basketball': '08', 'bow': '10', 'cannon': '12',
    'wheel': '13', 'catamaran': '16', 'chest': '18', 'coffee_mug': '20',
    'coffeepot': '21', 'hat': '23', 'crate': '25', 'desk': '27',
    'telephone': '28', 'disk_brake': '29', 'electric_locomotive': '30',
    'folding_chair': '32', 'pan': '34', 'piano': '37', 'horse_cart': '39',
    'jeep': '41', 'cover': '44', 'scooter': '49', 'bike': '50',
    'racer': '64', 'control': '67', 'revolver': '68', 'shoe': '70',
    'sofa': '75', 'lamp': '77', 'teapot': '79', 'toaster': '80',
    'toilet': '81', 'cleaner': '84', 'vase': '85', 'sign': '88',
    'traffic_light': '89'
}

# 你的ImageNet子集所在的根目录
imagenet_root = '/hy-tmp/imagenet'

print("开始重命名 ImageNet 子集文件夹...")

# 遍历 train 和 val 两个子目录
for split in ['train', 'val']:
    split_path = os.path.join(imagenet_root, split)
    if not os.path.isdir(split_path):
        print(f"警告: 目录 '{split_path}' 不存在，跳过。")
        continue
    
    print(f"--- 正在处理: {split_path} ---")
    for old_name in os.listdir(split_path):
        old_path = os.path.join(split_path, old_name)
        if os.path.isdir(old_path):
            if old_name in NAME_TO_ID_MAP:
                new_name = NAME_TO_ID_MAP[old_name]
                new_path = os.path.join(split_path, new_name)
                
                # 检查新路径是否已存在，避免重复运行出错
                if os.path.exists(new_path):
                    print(f"跳过: '{new_path}' 已存在。")
                else:
                    print(f"重命名: '{old_path}' -> '{new_path}'")
                    os.rename(old_path, new_path)
            else:
                print(f"警告: 在映射表中找不到文件夹 '{old_name}' 的对应ID，已跳过。")

print("\n--- 重命名完成！ ---")
print("现在你的 ImageNet 子集文件夹已按数字ID命名，与项目逻辑对齐。")