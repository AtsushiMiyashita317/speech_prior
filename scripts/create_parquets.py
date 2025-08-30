import json
from os import path
from pathlib import Path
import polars as pl
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def extract_info_from_meta(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    audio_path = Path(meta["audio_path_rel"])
    utterance_id = audio_path.stem
    return utterance_id, meta


def collect_meta_files(dumps_root, allowed_subsets):
    # dumps/<subset> 配下のみ探索。allowed_subsets指定時はそのみ、未指定なら全サブディレクトリ
    root = Path(dumps_root)
    subsets = allowed_subsets if allowed_subsets else [d.name for d in root.iterdir() if d.is_dir()]
    all_jsons = []
    for subset in subsets:
        subset_dir = root / subset
        if subset_dir.is_dir():
            all_jsons.extend(list(subset_dir.rglob("*.meta.json")))
    return all_jsons

def process_meta(meta_path, dumps_root):
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        audio_path = Path(meta["audio_path_rel"])
        rel_parts = Path(meta_path).relative_to(dumps_root).parts
        input_subset = rel_parts[1] if len(rel_parts) > 1 else None
        utterance_id = audio_path.stem
        manifest_row = {
            "utterance_id": utterance_id,
            "input_subset": input_subset,
            "audio_path_rel": meta.get("audio_path_rel"),
            "audio_sha256": meta.get("audio_sha256"),
            "sample_rate": meta.get("sample_rate"),
            "num_samples": meta.get("num_samples"),
            "duration_sec": meta.get("duration_sec"),
            "frontend": meta.get("frontend"),
        }
        index_row = {
            "utterance_id": utterance_id,
            "model_id": meta.get("model_id"),
            "layer": meta.get("layer"),
            "arch": meta.get("arch"),
            "dtype_saved": meta.get("dtype_saved"),
            "dtype_compute": meta.get("dtype_compute"),
            "shape": meta.get("shape"),
            "fps": meta.get("fps"),
            "frame_hop_ms": meta.get("frame_hop_ms"),
            "subsampling": meta.get("subsampling"),
            "hop_samples": meta.get("hop_samples"),
            "window_samples": meta.get("window_samples"),
            "frontend": meta.get("frontend"),
            "dump_path_rel": meta.get("dump_path_rel"),
            "audio_path_rel": meta.get("audio_path_rel"),
            "audio_sha256": meta.get("audio_sha256"),
            "script_version": meta.get("script_version"),
            "framework_versions": meta.get("framework_versions"),
            "device": meta.get("device"),
            "created_at": meta.get("created_at"),
            "system": meta.get("system"),
        }
        return manifest_row, index_row
    except Exception as e:
        print(f"Error processing {meta_path}: {e}")
        return None, None

def main(dumps_root, manifest_path, index_path, allowed_subsets):
    meta_files = collect_meta_files(dumps_root, allowed_subsets)
    manifest_rows = []
    index_rows = []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda p: process_meta(p, dumps_root), meta_files), total=len(meta_files), desc="Processing meta.json files"))
    for manifest_row, index_row in results:
        if manifest_row: manifest_rows.append(manifest_row)
        if index_row: index_rows.append(index_row)
    manifest_df = pl.DataFrame(manifest_rows)
    manifest_df.write_parquet(manifest_path)
    index_df = pl.DataFrame(index_rows)
    index_df.write_parquet(index_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create manifest.parquet and index.parquet from meta.json files under dumps root.")
    parser.add_argument("dumps_root", type=str, help="Path to dumps directory (e.g. dumps)")
    parser.add_argument("manifest_path", type=str, help="Output manifest Parquet file path")
    parser.add_argument("index_path", type=str, help="Output index Parquet file path")
    parser.add_argument("--allowed_subsets", type=str, nargs="*", help="List of allowed input subsets")
    args = parser.parse_args()
    main(args.dumps_root, args.manifest_path, args.index_path, args.allowed_subsets)
