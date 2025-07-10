#!/bin/bash
#SBATCH --partition=a100_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=4 
#SBATCH --time=14-00:00:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=TGVN_6x_3x
#SBATCH --output=TGVN_6x_3x.out

export OMP_NUM_THREADS=4
torchrun --nproc_per_node=4 --nnodes=$SLURM_JOB_NUM_NODES ../scripts/main_fastmri.py --acc-p 3 --center-freq-p 0 --acc-s 6 --center-freq-s 0.05 --num-casc 14 --num-chans 21 --type tgvn
