#!/bin/bash

#SBATCH --partition=physical

#SBATCH --job-name="prepare"
#SBATCH --account=punim1623
#SBATCH --time=0-02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

### SBATCH --mail-user=haitianh@student.unimelb.edu.au
### SBATCH --mail-type=BEGIN

module purge

eval "$(conda shell.bash hook)"
conda activate frust

python3 datasets/prepare_mvtec_few_shot_style_voc.py

##Log this job's resource usage stats###
my-job-stats -a -n -s
##