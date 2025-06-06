#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=6:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_ssd=40gb
#PBS -N diplomka


CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF

ls /cvmfs/singularity.metacentrum.cz

singularity run --nv $CONTAINER bash -c "
cd /auto/plzen1/home/dschnurp/dip/  && \
set PIP_NO_INPUT && \
pip install torch && \
pip install ultralytics && \
pip install huggingface-hub  && \
pip install opencv-python-headless==4.8.1.78 && \
pip install gcd && \
export WANDB_MODE=disabled && \
python /auto/plzen1/home/dschnurp/dip/fine_tune_yolo.py --model $model --epochs $epochs --project $thistime --data $data --conf-thres $confthres --freeze $freeze
"
