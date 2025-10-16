
runing_file=/home2/zhangyan/codes/wu/MLL-Theory/training/mll_on_text_data.py


CUDA_VISIBLE_DEVICES='2' python $runing_file --swa --loss_name u4 --lr 1.0e-4 --weight_decay 0.1 --config amazoncat13k/amazoncat13k-bert.yaml
CUDA_VISIBLE_DEVICES='2' python $runing_file --swa --loss_name u4 --lr 1.0e-4 --weight_decay 0. --config amazoncat13k/amazoncat13k-bert.yaml
CUDA_VISIBLE_DEVICES='2' python $runing_file --swa --loss_name u4 --lr 1.0e-5 --weight_decay 0.001 --config amazoncat13k/amazoncat13k-bert.yaml
CUDA_VISIBLE_DEVICES='2' python $runing_file --swa --loss_name u4 --lr 1.0e-5 --weight_decay 0.01 --config amazoncat13k/amazoncat13k-bert.yaml
