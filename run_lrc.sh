#!/bin/bash

# 可用GPU列表
GPUS=(4 5)
NUM_GPUS=${#GPUS[@]}  # GPU数量
GPU_IDX=0             # 当前使用到第几张GPU

# # delicious
# GPUS=(4 5)
# NUM_GPUS=${#GPUS[@]}  # GPU数量
# GPU_IDX=4             # 当前使用到第几张GPU

# 固定参数s
# SEED=100
# BATCH_SIZE=256
# NUM_EPOCH=400
# NUM_VAL=1000
# STEP=100
# GAMMA=0.1
# lr=0.0001
# weight_decay=0.0001
#epoch=30
#lr=0.05
#n_hidden=256
# 可调参部分
datasets=("goemotions") #("yeast" "corel5k") # "rcv1subset1_top944" "bibtex")
# svd_rate_list=(70 80 90) #20 30 40 50 60 70 80 90)
weight_decay_list=(0.001)
lr_list=(0.0001)
# svd_lambda_list=(0.1 0.01 0.001 0.00001 0.000001)

# datasets=("delicious")
# weight_decay_list=(0.1 0.01 0.001 0.0001)
# svd_lambda_list=(0.1) # 0.01 0.001 0.00001 0.000001


# 遍历参数组合
for dataset in "${datasets[@]}"
do
#   if [ "$dataset" = "CAL500" ]; then
#     svd_rate=80
#   else
#     svd_rate=100
#   fi
  for weight_decay in "${weight_decay_list[@]}"
  do
    # for svd_lambda in "${svd_lambda_list[@]}"
    for lr in "${lr_list[@]}"
    do
      # for svd_rate in "${svd_rate_list[@]}"
      # do
        # 选GPU
        CURRENT_GPU=${GPUS[$GPU_IDX]}

        # 生成log文件名
        # log_file="datasets${corruption_level}_weights${imbalance_rate_train}_svds${imbalance_rate_test}.log"
        # log_file="lrc_logs/datasets_${dataset}_weight_decay_${weight_decay}_svd_lambda_${svd_lambda}_svd_rate_${svd_rate}.log"
        log_file="logs_emotions/datasets_${dataset}_weight_decay_${weight_decay}_lr_${lr}_500.log"

        echo "使用GPU${CURRENT_GPU} 开始训练: dataset=$dataset, weight_decay=$weight_decay, lr=$lr"

        # 启动训练
        CUDA_VISIBLE_DEVICES=${CURRENT_GPU} nohup /home/public/anaconda/anaconda/envs/macro-auc-theory1/bin/python pa_linear_model_large.py \
        --dataset=${dataset} \
        --weight_decay=${weight_decay} \
        --lr=${lr} \
        > ${log_file} 2>&1 &

        # --svd_rate=${svd_rate} \ 
        # --svd_lambda=${svd_lambda} \

        # 更新 GPU_IDX，循环使用
        GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))

        sleep 2  # 适当休眠，避免起进程太快
      # done
    done
  done
done
