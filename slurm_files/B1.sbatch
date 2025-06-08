#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03-00:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=TGVN_18x_1x
#SBATCH --output=TGVN_18x_1x.out

export OMP_NUM_THREADS=4
torchrun --nproc_per_node=4 --nnodes=$SLURM_JOB_NUM_NODES ../scripts/main_m4.py --acc 32 --center-freq 0.02 --num-casc 10 --num-chans 21 --type tgvn
