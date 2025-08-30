#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=6:00:00
#PJM -j

subsets=(
    train-clean-360
    train-other-500
)

cd ${DATA}/gitrepo/speech_prior/

# データ保存先を用意
    mkdir -p data/librispeech
    cd data/librispeech

for subset in "${subsets[@]}"; do
    echo "Downloading and extracting ${subset}..."
    

    wget http://www.openslr.org/resources/12/${subset}.tar.gz
    tar -xvzf ${subset}.tar.gz
done
