#!/bin/bash

export PYTHONPATH="${PWD}"

python viat/train_trades_imagenet_viewpoint_new.py \
    --root_dir 'datasets/GMFool_dataset/airliner_01' \
    --ckpt_path 'run_train_nerf/ckpts/nerf/train/00/00.ckpt' \
    --ckpt_attack_path 'run_train_nerf/ckpts/nerf' \
    --epochs 2 \
    --batch-size 32 \
    --test-batch-size 32 \
    --iteration 10 \
    --iteration_warmstart 5 \
    --popsize 21 \
    # --save_freq 1 \  <-- 删掉这一行
    --treat_model 'resnet50' \
    --AT_exp_name 'minibatch_test_run' \
    --dataset_name nerf_for_attack \
    --train_mood 'AT' \
    --AT_type 'AVDT' \
    --num_k 5 \
    --share_dist \
    --share_dist_rate 0.5 \
    --no_background