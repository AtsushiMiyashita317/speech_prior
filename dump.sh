#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j

subsets=(
    dev-other
)

export HF_HUB_DISABLE_XET=1

cd ${DATA}/gitrepo/speech_prior/
source venv/bin/activate

pids=()  # BEGIN: Initialize an array to hold PIDs

for subset in "${subsets[@]}"; do
    touch ${subset}_hubert.log
    python scripts/dump_features.py \
    --input_dir data/librispeech/LibriSpeech/${subset} \
    --outdir dumps \
    --exts .flac \
    --arch hubert \
    --model facebook/hubert-base-ls960 \
    --layers 6,9,11 > ${subset}_hubert.log 2>&1 &
    pid=$!
    pids+=($pid)  # Store the PID
    echo "Process ID: $pid"

    touch ${subset}_wav2vec.log
    python scripts/dump_features.py \
    --input_dir data/librispeech/LibriSpeech/${subset} \
    --outdir dumps \
    --exts .flac \
    --arch wav2vec \
    --model facebook/wav2vec2-base \
    --layers 6,9,11 > ${subset}_wav2vec.log 2>&1 &
    pid=$!
    pids+=($pid)  # Store the PID
    echo "Process ID: $pid"

    touch ${subset}_wavlm.log
    python scripts/dump_features.py \
    --input_dir data/librispeech/LibriSpeech/${subset} \
    --outdir dumps \
    --exts .flac \
    --arch wavlm \
    --model microsoft/wavlm-base \
    --layers 6,9,11 > ${subset}_wavlm.log 2>&1 &
    pid=$!
    pids+=($pid)  # Store the PID
    echo "Process ID: $pid"

    
    touch ${subset}_whisper.log
    python scripts/dump_features.py \
    --input_dir data/librispeech/LibriSpeech/${subset} \
    --outdir dumps \
    --exts .flac \
    --arch whisper \
    --model openai/whisper-small \
    --layers 6,9,11 > ${subset}_whisper.log 2>&1 &
    pid=$!
    pids+=($pid)  # Store the PID
    echo "Process ID: $pid"

    touch ${subset}_xvector.log
    python scripts/dump_features.py \
    --input_dir data/librispeech/LibriSpeech/${subset} \
    --outdir dumps \
    --exts .flac \
    --arch xvector \
    --model speechbrain/spkrec-xvect-voxceleb \
    --layers 1,2,3 > ${subset}_xvector.log 2>&1 &
    pid=$!
    pids+=($pid)  # Store the PID
    echo "Process ID: $pid"
done

for pid in "${pids[@]}"; do
    wait $pid
done
