#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg1

#SBATCH --job-name="surgery"
#SBATCH --account=punim1623
#SBATCH --time=0-00:10:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=60G

module purge

eval "$(conda shell.bash hook)"
conda activate frust

python3 -m tools.ckpt_surgery \
	--src1 checkpoints/mvtec/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth \
	--method remove \
	--save-dir checkpoints/mvtec/faster_rcnn/faster_rcnn_R_101_FPN_all \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##

####  python3 -m tools.train_net --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml --opts MODEL.DEVICE cpu