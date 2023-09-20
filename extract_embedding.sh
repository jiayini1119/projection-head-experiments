#!/bin/bash
GPUS=("0" "1" "2" "3" "4" "5" "6" "7")

USE_PH=false
USE_PREV_BLOCK=false

DATASET_DIRS=(
    "/home/jennyni/datasets/imagenet-r"
    "/home/jennyni/datasets/imagenet-a"
    "/home/jennyni/datasets/bg_challenge/original"
    "/home/jennyni/datasets/bg_challenge/only_fg"
    "/home/jennyni/datasets/bg_challenge/mixed_rand"
)

DATASET_NAMES=(
    "imagenet-r"
    "imagenet-a"
    "bg_challenge"
    "bg_challenge"
    "bg_challenge"
)

for i in "${!DATASET_DIRS[@]}"; do
    if [ "$USE_PREV_BLOCK" = true ]; then
        python imagenet_extract_embeddings.py --batch_size=100 --split val --dataset_dir ${DATASET_DIRS[$i]} --dataset ${DATASET_NAMES[$i]} --device ${GPUS[$i]} --use_prev_block &
    else
        if [ "$USE_PH" = false ]; then
            python imagenet_extract_embeddings.py --batch_size=100 --split val --dataset_dir ${DATASET_DIRS[$i]} --dataset ${DATASET_NAMES[$i]} --device ${GPUS[$i]} &
        else
            python imagenet_extract_embeddings.py --batch_size=100 --split val --dataset_dir ${DATASET_DIRS[$i]} --dataset ${DATASET_NAMES[$i]} --device ${GPUS[$i]} --use-ph &
        fi
    fi
done

wait
