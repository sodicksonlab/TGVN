# Project Repository

This repository contains code and scripts for training and validating TGVNs to replicate https://arxiv.org/abs/2501.03021. Below is an overview of the files and their purposes. As the repository was built on fastMRI, the requirements can be found on https://github.com/facebookresearch/fastMRI

## File Descriptions

### Data and Splits
- **`fastmri_split`**: Contains the data split for the fastMRI knee dataset. Note that the csv files contain absolute paths, so the user should modify them depending on the dataset location. 
- **`m4raw_split`**: Contains the data split for the M4Raw dataset. Note that the csv files contain absolute paths, so the user should modify them depending on the dataset location.

### SLURM Batch Scripts
- **`Set_I.sbatch`**: SLURM script for running the first set of experiments using TGVN.
- **`Set_II.sbatch`**: SLURM script for running the second set of experiments using TGVN.
- **`Set_III.sbatch`**: SLURM script for running the third set of experiments using TGVN.

### Core Code Files
- **`custom_losses.py`**: Implements the custom loss function for training models.
- **`data.py`**: Contains data loading and preprocessing logic for fastMRI knee and M4Raw datasets.
- **`distributed.py`**: Handles distributed training setup and utilities.
- **`main_fastmri.py`**: Main script for training and evaluating models on the fastMRI dataset.
- **`main_m4.py`**: Main script for training and evaluating models on the M4Raw dataset.
- **`models.py`**: Defines TGVN architecture used in the project.

### Running Experiments
1. Update the SLURM batch scripts (`Set_I.sbatch`, `Set_II.sbatch`, `Set_III.sbatch`) with the appropriate parameters for your environment.
2. Submit jobs using:
   ```bash
   sbatch Set_I.sbatch
   sbatch Set_II.sbatch
   sbatch Set_III.sbatch
   ```

## Questions
For any questions or issues, feel free to reach out or open an issue in this repository.
