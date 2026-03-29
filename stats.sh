#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j


cd ${DATA}/gitrepo/speech_prior/
source venv/bin/activate

python scripts/collect_stats.py --subset train-clean-100 --output stats/train-clean-100.npz 