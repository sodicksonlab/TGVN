# Project Repository Overview

This repository contains code and scripts for training and validating **T**rust **G**uided **V**ariational **N**etworks (**TGVN**) to replicate https://arxiv.org/abs/2501.03021. Below is an overview of the files and their purposes. The repo was built on the fastMRI repo, which is accessible at https://github.com/facebookresearch/fastMRI. While this codebase utilizes only PyTorch, any wrapper can be integrated easily.

For convenience, a devcontainer that supports CUDA acceleration (CUDA 12.8) was added. For other CUDA versions and ROCm, lines 18 and 19 in the Dockerfile needs to be modified according to https://pytorch.org/get-started. The repo follows the standard layout, so the package can be installed with `pip install -e .` from the repo root. 

## File Descriptions

### Data and Splits
* The fastMRI dataset can be downloaded at https://fastmri.med.nyu.edu
* The M4Raw dataset (version 1.6) can be downloaded at https://zenodo.org/records/8056074

The dataset splits are provided as CSV files under `data_splits`.
- **`data_splits/fastmri`**: Contains the training, validation and test splits for the fastMRI knee dataset. Note that the fastMRI training and validation sets were combined and the training/validation/test splits for our experiments were created on a per-patient basis. The CSV files contain filenames, so the user should modify them depending on the dataset location.
- **`data_splits/m4raw`**: Contains the training, validation and test splits for the M4Raw dataset. Note that the CSV files contain split and filenames, so the user should modify them depending on the dataset location.

### SLURM Batch Scripts
- **`K1.sbatch`**: SLURM script for running the experiment K1 using TGVN / E2E-VarNet.
- **`K2.sbatch`**: SLURM script for running the experiment K2 using TGVN / E2E-VarNet.
- **`K3.sbatch`**: SLURM script for running the experiment K3 using TGVN / E2E-VarNet.
- **`B1.sbatch`**: SLURM script for running the experiment B1 using TGVN / E2E-VarNet.
- **`B2.sbatch`**: SLURM script for running the experiment B2 using TGVN / E2E-VarNet.

### Core Code Files
- **`scripts/main_fastmri.py`**: Main script for training and evaluating models on the fastMRI dataset.
- **`scripts/main_m4.py`**: Main script for training and evaluating models on the M4Raw dataset.
- **`src/tgvn/loss.py`**: Implements the MS-SSIM-L1 loss function for training models.
- **`src/tgvn/data.py`**: Contains data loading and preprocessing logic for fastMRI knee and M4Raw datasets.
- **`src/tgvn/distributed.py`**: Handles distributed training setup and utilities for multi-GPU, multi-node training.
- **`src/tgvn/models.py`**: Defines the TGVN architecture used in the project.

### Running Experiments
1. Update the SLURM batch scripts (`K1.sbatch`, `K2.sbatch`, `K3.sbatch`, `B1.sbatch`, `B2.sbatch`) with the appropriate parameters for your environment. The names of the sbatch files correspond to the experiments in the article. For details, see Table 1.
2. Submit jobs using:
   ```bash
   sbatch K1.sbatch
   sbatch K2.sbatch
   sbatch K3.sbatch
   sbatch B1.sbatch
   sbatch B2.sbatch
   ```

## P.S.
The M4Raw dataset contains multiple repetitions. We used the undersampled first repetition as input and the averaged RSS images (averaged over the repetition dimension) as the ground truth, which required only minimal HDF5 file manipulation.
For example, if `file_T101.h5`, `file_T102.h5`, `file_T103.h5` contain data from three repetitions for a given patient, we retained the k-space from `file_T101.h5` and computed the ground truth by averaging the RSS images. The resulting file was saved as `file_T1.h5`.

In the brain experiments, the overall acceleration factors differ from the undersampling factors due to repetition reduction. Specifically, for B1, $18\times$ undersampling combined with $2\times$ repetition reduction results in a $36\times$ practical acceleration. For B2, $15\times$ undersampling and $3\times$ repetition reduction yield a $45\times$ practical acceleration.

## Questions
For any questions or issues, feel free to reach out or open an issue in this repository.
