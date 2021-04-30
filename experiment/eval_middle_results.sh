#!/bin/bash

dataset=(
    'None'                # dummy
    'magazines'
    'apparel'
    'health_personal_care'
    'camera_photo'
    'toys_games'
    'software'
    'baby'
    'kitchen_housewares'
    'sports_outdoors'
    'electronics'
    'books'
    'video'
    'imdb'
    'dvd'
    'music'
    'MR'
)

GPU_ID=0
approach='IPRLS'
setting='Scrach'

for TASK_ID in `seq 1 16`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python ./inference.py \
        --approach $approach \
        --dataset ${dataset[TASK_ID]} --num_classes 2 \
        --load_folder checkpoints/IPRLS/$setting/1/${dataset[TASK_ID]}/prune \
        --mode prune \
        --log_path logs/Amazon_inference.log
done
