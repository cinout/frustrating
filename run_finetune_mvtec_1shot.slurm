#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg1

#SBATCH --job-name="finetune"
#SBATCH --account=punim1623
#SBATCH --time=0-03:00:00

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
	--config-file configs/MVTEC-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
	--opts MODEL.WEIGHTS checkpoints/mvtec/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth

##Log this job's resource usage stats###
my-job-stats -a -n -s
##