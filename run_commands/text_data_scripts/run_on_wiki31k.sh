#!/bin/bash
runing_file=/home2/zhangyan/codes/wu/MLL-Theory/training/mll_on_text_data.py


CUDA_VISIBLE_DEVICES='1' $py $runing_file --swa --loss_name u1 --lr 1.0e-4 --weight_decay 0.1 --config wiki31k/wiki31k-bert.yaml
CUDA_VISIBLE_DEVICES='1' $py $runing_file --swa --loss_name u2 --lr 1.0e-3 --weight_decay 0.1 --config wiki31k/wiki31k-bert.yaml
CUDA_VISIBLE_DEVICES='1' $py $runing_file --swa --loss_name u3 --lr 1.0e-5 --weight_decay 0.1 --config wiki31k/wiki31k-bert.yaml
CUDA_VISIBLE_DEVICES='1' $py $runing_file --swa --loss_name u4 --lr 1.0e-4 --weight_decay 0.1 --config wiki31k/wiki31k-bert.yaml

