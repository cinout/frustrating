#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg1

#SBATCH --job-name="train_base"
#SBATCH --account=punim1623
#SBATCH --time=0-10:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=60G

module purge

eval "$(conda shell.bash hook)"
conda activate frust

python3 -m tools.train_net --num-gpus 4 \
	--config-file configs/MVTEC-detection/faster_rcnn_R_101_FPN_base.yaml

##Log this job's resource usage stats###
my-job-stats -a -n -s
##

####  python3 -m tools.train_net --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml --opts MODEL.DEVICE cpu