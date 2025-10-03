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
    --root_dir '/hy-tmp/GMFool_dataset/airliner_01' \
    --ckpt_path '/hy-tmp/VIAT_outputs/ckpts/nerf/train/01/00.ckpt' \
    --ckpt_attack_path '/hy-tmp/VIAT/run_train_nerf/ckpts/nerf' \
    --output_dir '/hy-tmp/VIAT_outputs' \
    --dataset_name nerf_for_attack \
    --scene_name 'attack_on_resnet50_k5' \
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