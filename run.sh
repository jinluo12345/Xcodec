#!/usr/bin/env bash

#SBATCH -p fnlp-4090d
#SBATCH --job-name=xcodec-lzjjin
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=12G

#SBATCH --output=/remote-home1/lzjjin/X-Codec-2.0/output.log

source ~/.bashrc
conda init
conda activate xcodec
which python

work_dir=/remote-home1/lzjjin/X-Codec-2.0
cd ${work_dir}
export PYTHONPATH="${work_dir}"

# Exps

cmd="python train.py  train.trainer.accelerator=gpu train.trainer.devices=8"

echo "Executing: ${cmd}"
eval ${cmd}