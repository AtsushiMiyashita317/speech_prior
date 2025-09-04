import os
import glob
import numpy as np
import h5py
import polars as pl
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torchaudio
from pathlib import Path

def find_utt_dirs(dumps_root, subsets=[]):
    # dumps/**/<utt_id> ディレクトリを再帰的に探索
    utt_dirs = []
    if subsets:
        for subset in subsets:
            subset_root = os.path.join(dumps_root, subset)
            for root, dirs, files in os.walk(subset_root):
                for d in dirs:
                    utt_path = os.path.join(root, d)
                    npy_files = glob.glob(os.path.join(utt_path, "*.npy"))
                    if npy_files:
                        utt_dirs.append(utt_path)
        return utt_dirs
    
    for root, dirs, files in os.walk(dumps_root):
        for d in dirs:
            utt_path = os.path.join(root, d)
            npy_files = glob.glob(os.path.join(utt_path, "*.npy"))
            if npy_files:
                utt_dirs.append(utt_path)
    return utt_dirs

def process_utt_dir(utt_dir, raw_root, dumps_root, dataset_root):  
    # utt_id ディレクトリ内の全npyをロードし、(T, D_total)に連結
    npy_files = sorted(glob.glob(os.path.join(utt_dir, "*.npy")))
    arrays = []
    col_start = 0
    features_records = []
    for fpath in npy_files:
        arr = np.load(fpath)
        arrays.append(arr)
        D = arr.shape[1]
        key = os.path.splitext(os.path.basename(fpath))[0]  # <model>.<layer>
        start = col_start
        end = col_start + D
        col_start += D
        model, layer = key.split('.', 1)
        features_records.append({
            "model": model,
            "layer": layer,
            "D": D,
            "start": start,
            "end": end
        })
    T = arrays[0].shape[0]
    for arr in arrays:
        assert arr.shape[0] == T, f"T mismatch in {utt_dir}, shapes: {[a.shape for a in arrays]}"
    data = np.concatenate(arrays, axis=1)  # (T, D_total)

    # dataset/**/<utt_id>.h5 のパスを決定
    rel_path = os.path.relpath(utt_dir, dumps_root)
    out_dir = os.path.join(dataset_root, os.path.dirname(rel_path))
    os.makedirs(out_dir, exist_ok=True)
    utt_id = os.path.basename(utt_dir)
    out_path = os.path.join(out_dir, f"{utt_id}.h5")

    rel_utt_dir = os.path.relpath(utt_dir, dumps_root)
    subset = Path(rel_utt_dir).parts[0]
    
    wav_path = os.path.join(raw_root, rel_utt_dir + ".flac")
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2:
        if wav.shape[0] == 1:
            wav = wav[0]
        else:
            wav = wav.mean(dim=0)
    wav = wav.numpy().astype(np.float16)
    
    with h5py.File(out_path, "w") as f:
        group = f.create_group("utterance")
        group.create_dataset("waveform", data=wav)
        group.create_dataset("features", data=data)

    
    utterance_record = {
        "utt_id": utt_id,
        "subset": subset,
        "T": T,
        "path": rel_utt_dir
    }
    return features_records, utterance_record

def main():
    parser = argparse.ArgumentParser(description="Create h5 files and features/utterance parquet from dumps directory.")
    parser.add_argument("--raw_root", type=str, default="data/librispeech/LibriSpeech", help="Path to raw directory")
    parser.add_argument("--dumps_root", type=str, default="dumps", help="Path to dumps directory")
    parser.add_argument("--dataset_root", type=str, default="datasets", help="Path to output dataset directory")
    parser.add_argument("--features_parquet", type=str, default="features.parquet", help="Output features parquet file")
    parser.add_argument("--utterance_parquet", type=str, default="utterance.parquet", help="Output utterance parquet file")
    parser.add_argument("--subsets", type=str, nargs='*', help="Subsets to process (e.g., train, dev, test). If not set, process all.")
    args = parser.parse_args()

    utt_dirs = find_utt_dirs(args.dumps_root, args.subsets)
    features_records = []
    utterance_records = []
    seen_features = set()
    def wrapper(utt_dir):
        return process_utt_dir(utt_dir, args.raw_root, args.dumps_root, args.dataset_root)
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(wrapper, utt_dirs), total=len(utt_dirs), desc="Processing utt_dirs"))
    for features, utterance in results:
        for feat in features:
            key = (feat["model"], feat["layer"])
            if key not in seen_features:
                features_records.append(feat)
                seen_features.add(key)
        utterance_records.append(utterance)

    # --- parquet保存 ---
    features_df = pl.DataFrame(features_records)
    utterance_df = pl.DataFrame(utterance_records)
    features_df.write_parquet(os.path.join(args.dataset_root, args.features_parquet))
    utterance_df.write_parquet(os.path.join(args.dataset_root, args.utterance_parquet))

if __name__ == "__main__":
    main()