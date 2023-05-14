#!/bin/bash
#SBATCH --job-name=CodeBERT_4GPU_Update
#SBATCH --partition=rtx8000
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --output=%x.result
#SBATCH --mem=128GB
#SBATCH --time=24:00:00

module load python/intel/3.8.6
module load anaconda3/2020.07
module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11

cd /scratch/kv2154/CodeBERT-master/CodeBERT/code2nl
eval "$(conda shell.bash hook)"
conda activate ../../env/

python3 -m pip install torch torchvision transformers filelock

nvidia-smi
lscpu
lshw
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,gpu_name,gpu_bus_id,compute_mode,fan.speed,pstate,clocks.current.graphics,clocks.current.memory,uuid --format=csv -l 1 > results-file-CodeBERT-4gpu_Update.csv &
NVIDIA_SMI_PID=$!

lang=python #programming language
lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../code2nl/CodeSearchNet
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps 
kill $NVIDIA_SMI_PID