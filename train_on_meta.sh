#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=3:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_ssd=40gb
#PBS -N diplomka


CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.04-py3.SIF

ls /cvmfs/singularity.metacentrum.cz

singularity run --nv $CONTAINER bash -c "
cd /auto/plzen1/home/dschnurp/dip/  && \
mkdir ./runs/$thistime && \
source /auto/plzen1/home/dschnurp/venv/bin/activate  && \
pip install -r requirements.txt --compile --no-cache-dir  && \
/auto/plzen1/home/dschnurp/dip/train_yolo.py --model $model --epochs $epochs  && \
cp -r ./runs /auto/plzen1/home/dschnurp/dip/runs/$thistime


