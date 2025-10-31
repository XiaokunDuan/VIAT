#!/bin/bash
#SBATCH --gpus=1

# 激活conda环境
# 注意: 如果您的conda环境不是全局的，可能需要先 source ~/.bashrc
# module load anaconda/2020.11 # 如果需要的话
source activate viat

# --- 关键步骤 1: 自动定位项目根目录 ---
# 获取脚本所在的目录
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# 项目根目录是脚本所在目录的上一级
PROJECT_ROOT_DIR=$(dirname "$SCRIPT_DIR")

# --- 关键步骤 2: 将项目根目录添加到PYTHONPATH ---
# 这会告诉Python解释器:"请到 $PROJECT_ROOT_DIR 目录下寻找模块"
export PYTHONPATH="$PROJECT_ROOT_DIR:$PYTHONPATH"

# --- 关键步骤 3: 切换到项目根目录 ---
# 这是一个好习惯，确保所有相对路径（如./results）都能正确工作
cd "$PROJECT_ROOT_DIR"

echo "PYTHONPATH 已设置为: $PYTHONPATH"
echo "当前工作目录: $(pwd)"
echo "开始运行攻击脚本..."

# 使用绝对路径运行Python脚本
python viat/Attack_exp_fast_K.py \
    --root_dir '/hy-tmp/GMFool_dataset/airliner_01' \
    --ckpt_path '/hy-tmp/VIAT_outputs/ckpts/nerf/train/01/00.ckpt' \
    --ckpt_attack_path '/hy-tmp/VIAT_outputs2/ckpts/nerf' \
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
    --treat_model 'resnet50' \
    --AT_exp_name 'k5_attack_experiment' \
    --num_k 5

echo "脚本运行结束。"