#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg1

#SBATCH --job-name="demo"
#SBATCH --account=punim1623
#SBATCH --time=0-00:10:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

module purge

eval "$(conda shell.bash hook)"
conda activate frust

python3 -m demo.demo --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
  --input mvtec_image_for_demo/breakfast_007.png mvtec_image_for_demo/juice_009.png mvtec_image_for_demo/pushpin_012.png mvtec_image_for_demo/screw_003.png mvtec_image_for_demo/splice_025.png \
  --output coco_1s_mvtec_demo_result \
  --opts MODEL.WEIGHTS pretrained_models/coco_tfa_cos_1shot_model_final.pth \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##