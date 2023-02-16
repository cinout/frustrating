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

python3 -m demo.demo --config-file configs/MVTEC-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
  --input mvtec_novel_images_for_demo/breakfast_box/079.png mvtec_novel_images_for_demo/breakfast_box/088.png mvtec_novel_images_for_demo/breakfast_box/097.png mvtec_novel_images_for_demo/breakfast_box/140.png mvtec_novel_images_for_demo/breakfast_box/223.png \
  --output mvtec_novel_demo_result \
  --opts MODEL.WEIGHTS checkpoints/mvtec/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all_1shot_randnovel/model_final.pth \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##