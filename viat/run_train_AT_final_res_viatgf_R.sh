#!/bin/bash
#SBATCH --gpus=1
#module load anaconda/2020.11
#source activate fastNeRF
source activate viat

# --- 自动设置项目根目录和PYTHONPATH ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT_DIR=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="$PROJECT_ROOT_DIR:$PYTHONPATH"
cd "$PROJECT_ROOT_DIR"

echo "PYTHONPATH 已设置为: $PYTHONPATH"
echo "当前工作目录: $(pwd)"
echo "开始运行 VIAT 防御训练脚本..."


# 1
python viat/train_trades_imagenet_viewpoint_new.py \
    --root_dir '/hy-tmp/GMFool_dataset/airliner_01' \
    --dataset_name nerf_for_attack \
    --scene_name 'resnet_GMM/hotdog' \
    --N_importance 64 \
    --ckpt_path '/hy-tmp/VIAT_outputs/ckpts/nerf/train/01/00.ckpt'\
    --optim_method NES \
    --search_num 6 \
    --popsize 11 \
    --iteration 1 \
    --iteration_warmstart 10 \
    --mu_lamba 0.05 \
    --sigma_lamba 0.05 \
    --omiga_lamba 0.05 \
    --num_sample 100 \
    --train_mood 'AT'\
    --batch-size 128 \
    --test-batch-size 128 \
    --AT_exp_name 'k5_attack_experiment_eval_test' \
    --lr 0.001 \
    --epochs 120 \
    --num_k 15 \
    --no_background \
    --fast_AVDT \
    --share_dist \
    --AT_type 'AVDT' \
    --share_dist_rate 0.5 \
    --ckpt_attack_path '/hy-tmp/VIAT_outputs2/ckpts/nerf' \
    --treat_model 'resnet50'\
    --save-freq 1\
    --model-dir '/hy-tmp/VIAT_outputs2/robust_models' # <-- 指定模型保存目录
# 上面修改了batch_size 因为对抗样本和干净样本放一起显存不够了 --batch-size 和--test-batch-size 从512-> 128 另外popsize 101-> 11, iteration 50-> 1 save-frq 是新增的

echo "脚本运行结束。"
