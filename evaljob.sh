#!/bin/bash
#SBATCH --job-name=CodeBERT_4GPU_Eval_Update
#SBATCH --partition=rtx8000
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --output=%x.result
#SBATCH --mem=128GB
#SBATCH --time=12:00:00

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
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,gpu_name,gpu_bus_id,compute_mode,fan.speed,pstate,utilization.encoder,utilization.decoder,clocks.current.graphics,clocks.current.memory,uuid --format=csv -l 1 > results-file-CodeBERT-4gpu_update.csv &
NVIDIA_SMI_PID=$!

lang=python #programming language
beam_size=10
batch_size=128
source_length=256
target_length=128
output_dir=model/$lang
data_dir=CodeSearchNet/
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
kill $NVIDIA_SMI_PID