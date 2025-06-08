#!/bin/bash
#SBATCH --partition=a100_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=4 
#SBATCH --time=14-00:00:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=TGVN_20x_2x
#SBATCH --output=TGVN_20x_2x.out

export OMP_NUM_THREADS=4
torchrun --nproc_per_node=4 --nnodes=$SLURM_JOB_NUM_NODES ../scripts/main_fastmri.py --acc-p 2 --center-freq-p 0 --acc-s 20 --center-freq-s 0.03 --num-casc 14 --num-chans 21 --type tgvn
