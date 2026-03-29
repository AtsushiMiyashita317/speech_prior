#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=4:00:00
#PJM -j

cd ${DATA}/gitrepo/speech_prior/
source venv/bin/activate

python mppca.py
