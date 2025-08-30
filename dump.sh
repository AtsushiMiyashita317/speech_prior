#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -j

subsets=(
    train-clean-360
)

models=(
    facebook/hubert-base-ls960
    facebook/wav2vec2-base
    microsoft/wavlm-base
)

export HF_HUB_DISABLE_XET=1

cd ${DATA}/gitrepo/speech_prior/
source venv/bin/activate

pids=()  # BEGIN: Initialize an array to hold PIDs

for subset in "${subsets[@]}"; do
    for model in "${models[@]}"; do
        model_name=${model//\//_}  # Replace / with _ in model name
        touch ${subset}_${model_name}.log
        python scripts/dump_features.py \
        --input_dir data/librispeech/LibriSpeech/${subset} \
        --outdir dumps \
        --root . \
        --exts .flac \
        --model "$model" \
        --layers 6,9,11 \
        --num_workers 1 > ${subset}_${model_name}.log 2>&1 &
        pid=$!
        pids+=($pid)  # Store the PID
        echo "Process ID: $pid"
    done

    touch ${subset}_whisper.log
    python scripts/dump_features.py \
    --input_dir data/librispeech/LibriSpeech/${subset} \
    --outdir dumps \
    --root . \
    --exts .flac \
    --arch whisper \
    --model openai/whisper-small \
    --layers 6,9,11 \
    --num_workers 1 > ${subset}_whisper.log 2>&1 &
    pid=$!
    pids+=($pid)  # Store the PID
    echo "Process ID: $pid"

    touch ${subset}_xvector.log
    python scripts/dump_features.py \
    --input_dir data/librispeech/LibriSpeech/${subset} \
    --outdir dumps \
    --root . \
    --exts .flac \
    --arch xvector \
    --model speechbrain/spkrec-xvect-voxceleb \
    --layers 1,2,3 \
    --num_workers 1 > ${subset}_xvector.log 2>&1 &
    pid=$!
    pids+=($pid)  # Store the PID
    echo "Process ID: $pid"
done

for pid in "${pids[@]}"; do
    wait $pid
done
