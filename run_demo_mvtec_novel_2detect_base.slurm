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

python3 -m demo.demo --config-file configs/MVTEC-detection/faster_rcnn_R_101_FPN_ft_all_5shot.yaml \
  --input mvtec_base_images_for_demo/2007_000332.jpg mvtec_base_images_for_demo/2007_000363.jpg mvtec_base_images_for_demo/2007_000663.jpg mvtec_base_images_for_demo/2007_000830.jpg mvtec_base_images_for_demo/2007_000837.jpg mvtec_base_images_for_demo/2007_001154.jpg mvtec_base_images_for_demo/2007_001457.jpg mvtec_base_images_for_demo/2007_001709.jpg mvtec_base_images_for_demo/2007_002260.jpg \
  --output mvtec_novel_2_base_demo_result \
  --opts MODEL.WEIGHTS checkpoints/mvtec/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all_5shot_randnovel/model_final.pth \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##