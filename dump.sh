subsets=(
    dev-clean
    train-clean-100
)

models=(
    facebook/hubert-base-ls960
    facebook/wav2vec2-base
    microsoft/wavlm-base
)

for subset in "${subsets[@]}"; do
    for model in "${models[@]}"; do
        python scripts/dump_features.py \
        --input_dir data/librispeech/LibriSpeech/${subset} \
        --outdir dumps \
        --root . \
        --exts .flac \
        --model "$model" \
        --layers 6,9,11 \
        --num_workers 1
    done

    python scripts/dump_features.py \
    --input_dir data/librispeech/LibriSpeech/${subset} \
    --outdir dumps \
    --root . \
    --exts .flac \
    --arch whisper \
    --model openai/whisper-small \
    --layers 6,9,11 \
    --num_workers 1

    python scripts/dump_features.py \
    --input_dir data/librispeech/LibriSpeech/${subset} \
    --outdir dumps \
    --root . \
    --exts .flac \
    --arch xvector \
    --model speechbrain/spkrec-xvect-voxceleb \
    --layers 1,2,3 \
    --num_workers 1
done
