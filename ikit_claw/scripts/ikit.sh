#!/bin/bash
cd ..
source venv/bin/activate
#export CUDA_LAUNCH_BLOCKING=1
python main.py --bsz=1 --vt_bsz=1 --device=-1
