# Road Segmentation for RoboTour 2024

This repository contains the code for training and testing a road segmentation model for the RoboTour 2024 competition.

## Installation and setup

The code is meant to be developed locally in a Python=3.10 environment and then run on a remote server. The following
instructions are for setting up the local development environment.

### Local setup

1. Clone the repository and navigate to the root directory of the project.
2. Install PyTorch by following the instructions on the [official website](https://pytorch.org/get-started/locally/).
3. Install other requirements by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

### Remote setup

Remote machine should use Slurm Workload Manager to manage jobs. The following steps are for setting up the remote

1. Clone the repository and navigate to the root directory of the project.
2. There is no need to install any requirements on the remote machine as the dependencies are loaded in the Slurm
   script.
   The script will load the modules required for the job.
3. Run the following command to submit a job to the Slurm queue:
   ```bash
   sbatch [sbatch options] train.batch [script options]
   ```

## Usage

The code is divided into two main parts: training and testing. To run training on the local machine, run the following
command:

   ```bash
   python main.py action=train ckpt_path=[checkpoint path] run_name=[run name]
   ```

To run testing on the local machine, run the following command:

   ```bash
   python main.py action=test ckpt_path=[checkpoint path] run_name=[run name]
   ```
