#!/bin/bash
# Normally, bash shell cannot support floating point arthematic, thus, here we use `bc` package

task_seed=1

python ./random_generate_task_sequence.py --seed $task_seed
readarray dataset < shuffle_task_order.txt
for task_id in `seq 1 16`; do
    dataset[$task_id]=`echo ${dataset[task_id]}`
done

echo "run with task order: ${dataset[*]}"

GPU_ID=0
setting='Random_task_order'

approach='IPRLS'
finetune_epochs=3
lr=5e-5
prune_lr=5e-5
num_classes=2
batch_size=32
total_num_tasks=16
seed=1

for task_id in `seq 1 16`; do

    if [ "$task_id" != "1" ]
    then
        CUDA_VISIBLE_DEVICES=$GPU_ID python ./main.py \
            --approach $approach \
            --dataset ${dataset[task_id]} --num_classes $num_classes \
            --seed $seed \
            --lr $lr \
            --batch_size $batch_size \
            --weight_decay 4e-5 \
            --save_folder checkpoints/$approach/$setting/$seed/${dataset[task_id]}/finetune \
            --load_folder checkpoints/$approach/$setting/$seed/${dataset[task_id-1]}/prune \
            --epochs $finetune_epochs \
            --mode finetune \
            --log_path checkpoints/$approach/$setting/$seed/${dataset[task_id]}/train.log \
            --total_num_tasks $total_num_tasks
    else
        CUDA_VISIBLE_DEVICES=$GPU_ID python ./main.py \
            --approach $approach \
            --dataset ${dataset[task_id]} --num_classes $num_classes \
            --seed $seed \
            --lr $lr \
            --batch_size $batch_size \
            --weight_decay 4e-5 \
            --save_folder checkpoints/$approach/$setting/$seed/${dataset[task_id]}/finetune \
            --epochs 3 \
            --mode finetune \
            --log_path checkpoints/$approach/$setting/$seed/${dataset[task_id]}/train.log \
            --total_num_tasks $total_num_tasks
    fi

    prune_epoch=3

    # Prune the model after training
    if [ "$task_id" == "1" ]
    then
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --approach $approach \
            --dataset ${dataset[task_id]} --num_classes $num_classes \
            --seed $seed \
            --lr $prune_lr \
            --batch_size $batch_size \
            --weight_decay 4e-5 \
            --save_folder checkpoints/$approach/$setting/$seed/${dataset[task_id]}/prune \
            --load_folder checkpoints/$approach/$setting/$seed/${dataset[task_id]}/finetune \
            --epochs $prune_epoch \
            --mode prune \
            --log_path checkpoints/$approach/$setting/$seed/${dataset[task_id]}/train.log \
            --total_num_tasks $total_num_tasks \
            --one_shot_prune_perc 0.4
    else
        echo $state
        # gradually pruning
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --approach $approach \
            --dataset ${dataset[task_id]} --num_classes $num_classes \
            --seed $seed \
            --lr $prune_lr \
            --batch_size $batch_size \
            --weight_decay 4e-5 \
            --save_folder checkpoints/$approach/$setting/$seed/${dataset[task_id]}/prune \
            --load_folder checkpoints/$approach/$setting/$seed/${dataset[task_id]}/finetune \
            --epochs $prune_epoch \
            --mode prune \
            --log_path checkpoints/$approach/$setting/$seed/${dataset[task_id]}/train.log \
            --total_num_tasks $total_num_tasks \
            --one_shot_prune_perc 0.75
    fi
done