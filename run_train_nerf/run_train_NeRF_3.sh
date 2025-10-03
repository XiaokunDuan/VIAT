#!/bin/bash
# #SBATCH --gpus=1
# module load anaconda/2020.11
# module load gcc/9.3
# module load CUDA/11.3.1
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate viat

# export CUDA_HOME=/data/apps/CUDA/11.3.1/
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
# export PATH=${CUDA_HOME}/bin:${PATH}
# export PATH
# export LD_LIBRARY_PATH=/data/apps/gcc/9.3.0/lib64:$LD_LIBRARY_PATH    


# python train.py --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/Synthetic_NeRF/Hotdog' --exp_name 'ngp_hotdog' --num_epochs 20 --batch_size 16384 --lr 2e-2

# barbell barrel basketball bike bow cannon catamaran
# 01 02 03 04 05 06 07 08

list=(28 29 30 32 34 37 39 41)
num=0

for j in telephone disk_brake electric_locomotive folding_chair pan piano horse_cart jeep
do 
    for i in 01 02 03 04 05 06 07 08 09
    do 
        PYTHONPATH=$PYTHONPATH:.. python ../viat/train.py \
            --dataset_name nerf \
            --root_dir ../../../../hy-tmp/GMFool_dataset/${j}_${i} \
            --exp_name new/train/${list[$num]} \
            --num_epochs 10 \
            --batch_size 16384 \
            --lr 1e-2
    done

    for i in 10
    do 
        PYTHONPATH=$PYTHONPATH:.. python ../viat/train.py \
            --dataset_name nerf \
            --root_dir ../../../../hy-tmp/GMFool_dataset/${j}_${i} \
            --exp_name new/test/${list[$num]} \
            --num_epochs 10 \
            --batch_size 16384 \
            --lr 1e-2
    done

    let num=$num+1
done




