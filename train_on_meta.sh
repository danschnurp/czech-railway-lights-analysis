#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_ssd=40gb
#PBS -N diplomka


CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.04-py3.SIF

ls /cvmfs/singularity.metacentrum.cz

singularity run --nv $CONTAINER cd /auto/plzen1/home/dschnurp/dip/
singularity run --nv $CONTAINER pip install ultralytics --user
singularity run --nv $CONTAINER rm -r /storage/praha1/home/dschnurp/.local/lib/python3.10/site-packages/cv2
singularity run --nv $CONTAINER python  train_yolo.py --model $model --epochs $epochs
