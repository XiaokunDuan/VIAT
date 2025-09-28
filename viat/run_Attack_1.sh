#!/bin/bash
#SBATCH --gpus=1

# 激活conda环境
# 注意: 如果您的conda环境不是全局的，可能需要先 source ~/.bashrc
# module load anaconda/2020.11 # 如果需要的话
source activate viat

# 切换到项目主目录，这是一个好习惯，可以避免很多相对路径问题
cd /hy-tmp/VIAT/

echo "当前工作目录: $(pwd)"
echo "开始运行攻击脚本..."

# 使用绝对路径运行Python脚本
python viat/Attack_exp_fast_K.py \
    \
    # ------------------- 数据集和NeRF模型相关路径 -------------------
    # 1. NeRF数据集的根目录 (用于加载相机参数，以airliner_01为例)
    --root_dir '/hy-tmp/VIAT/datasets/GMFool_dataset/airliner_01' \
    \
    # 2. 任意一个NeRF模型的检查点路径 (用于初始化)
    --ckpt_path '/hy-tmp/VIAT/run_train_nerf/ckpts/nerf/train/00/00.ckpt' \
    \
    # 3. [最关键!] 所有NeRF检查点的根目录 (指向train/test的父目录)
    --ckpt_attack_path '/hy-tmp/VIAT/run_train_nerf/ckpts/nerf' \
    \
    # ------------------- 脚本运行参数 -------------------
    --dataset_name nerf_for_attack \
    --scene_name 'results_resnet_GMM_hotdog' \
    --N_importance 64 \
    --optim_method NES \
    --search_num 6 \
    --popsize 101 \
    --iteration 50 \
    --iteration_warmstart 10 \
    --mu_lamba 0.05 \
    --sigma_lamba 0.05 \
    --omiga_lamba 0.05 \
    --num_sample 100 \
    --train_mood 'AT' \
    --batch-size 512 \
    --test-batch-size 512 \
    --lr 0.001 \
    --epochs 90 \
    --no_background \
    --share_dist \
    \
    # ------------------- 攻击目标和实验命名 -------------------
    # 4. 确认要攻击的目标模型名称
    --treat_model 'resnet50' \
    \
    # 5. 实验名称，用于保存结果
    --AT_exp_name 'k5_attack_experiment' \
    --num_k 5

echo "脚本运行结束。"