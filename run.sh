#!/bin/bash
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -t 0-01:00 
#SBATCH -p ampere 
#SBATCH --mem=40000 
#SBATCH -o output/a.o
#SBATCH -e output/a.e #
#SBATCH --job-name='hiera'
#SBATCH --gres=gpu:1 

nvidia-smi


export CUDA_VISIBLE_DEVICES=0
for s in 42
do
    python -u PTRN_1.py \
        --dataset pdtb3 \
        --task pdtb3 \
        --seed $s \
        --pretrain \
        --bert_model  '../../../model_dataset/roberta_base' \
        --data_dir '../data/pdtb3_duibi' \
        --log_name 'pdtb3-robert-base-contrastive-implicit' \
        --temperature 0.2 \
        --con1 1.6 \
        --con2 1.3 \
        --save_model_path 'pdtb3-robert-base-contrasive-implicit-1' \
        --train_batch_size 256 \
        --eval_batch_size 256 \
        --save_model \
        --num_train_epochs 25 \
        --wait_patient 10 \
        --dstore_size 20244 \
        --b1 2.0 \
        --lr 3e-05 \
        --warmup_proportion 0.1
done

        