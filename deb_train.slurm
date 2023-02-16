#!/bin/bash

#SBATCH --partition=gpu-a100

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg1

#SBATCH --job-name="deb_train"
#SBATCH --account=punim1623
#SBATCH --time=0-01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=60G

module purge

eval "$(conda shell.bash hook)"
conda activate frust

pypython3 -m tools.train_net --num-gpus 2 \
	--config-file configs/MVTEC-detection/faster_rcnn_R_101_FPN_base.yaml

##Log this job's resource usage stats###
my-job-stats -a -n -s
##