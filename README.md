# Project Repository

This repository contains code and scripts for training and validating TGVNs to replicate https://arxiv.org/abs/2501.03021. Below is an overview of the files and their purposes. As the repository was built on fastMRI, the requirements can be found on https://github.com/facebookresearch/fastMRI
Note that the M4Raw dataset contains multiple repetitions. We used the undersampled first repetition as input and the averaged RSS image as the ground truth, which required only minimal HDF5 file manipulation.
For example, if `file_T101.h5`, `file_T102.h5`, `file_T103.h5` contain data from three repetitions for a given patient, we retained the k-space from `file_T101.h5` and computed the ground truth by averaging the RSS images in the image domain. The resulting file was saved as `file_T1.h5`.
## File Descriptions

### Data and Splits
- **`fastmri_split`**: Contains the data split for the fastMRI knee dataset. Note that the csv files contain absolute paths, so the user should modify them depending on the dataset location. 
- **`m4raw_split`**: Contains the data split for the M4Raw dataset. Note that the csv files contain absolute paths, so the user should modify them depending on the dataset location.

### SLURM Batch Scripts
- **`K1.sbatch`**: SLURM script for running the experiment K1 using TGVN / E2E-VarNet.
- **`K2.sbatch`**: SLURM script for running the experiment K2 using TGVN / E2E-VarNet.
- **`K3.sbatch`**: SLURM script for running the experiment K3 using TGVN / E2E-VarNet.
- **`B1.sbatch`**: SLURM script for running the experiment B1 using TGVN / E2E-VarNet.
- **`B2.sbatch`**: SLURM script for running the experiment B2 using TGVN / E2E-VarNet.

### Core Code Files
- **`custom_losses.py`**: Implements the custom loss function for training models.
- **`data.py`**: Contains data loading and preprocessing logic for fastMRI knee and M4Raw datasets.
- **`distributed.py`**: Handles distributed training setup and utilities.
- **`main_fastmri.py`**: Main script for training and evaluating models on the fastMRI dataset.
- **`main_m4.py`**: Main script for training and evaluating models on the M4Raw dataset.
- **`models.py`**: Defines TGVN architecture used in the project.

### Running Experiments
1. Update the SLURM batch scripts (`K1.sbatch`, `K2.sbatch`, `K3.sbatch`, `B1.sbatch`, `B2.sbatch`) with the appropriate parameters for your environment.
2. Submit jobs using:
   ```bash
   sbatch K1.sbatch
   sbatch K2.sbatch
   sbatch K3.sbatch
   sbatch B1.sbatch
   sbatch B2.sbatch
   ```

## Questions
For any questions or issues, feel free to reach out or open an issue in this repository.
