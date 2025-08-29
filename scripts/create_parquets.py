
import json
from pathlib import Path
import polars as pl
import argparse
from tqdm import tqdm

def extract_info_from_meta(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    audio_path = Path(meta["audio_path_rel"])
    utterance_id = audio_path.stem
    return utterance_id, meta


def collect_meta_files(dumps_root):
    # dumps/<model_name>/<input_subset>/... の全meta.jsonを再帰的に探索
    return list(Path(dumps_root).rglob("*.json"))

def main(dumps_root, manifest_path, index_path):
    manifest_rows = []
    index_rows = []
    meta_files = collect_meta_files(dumps_root)
    for meta_path in tqdm(meta_files, desc="Processing meta.json files"):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            audio_path = Path(meta["audio_path_rel"])
            # input_subsetは dumps/<model>/<input_subset>/... の input_subset 部分
            # model_nameは dumps/<model>/... の model 部分
            rel_parts = Path(meta_path).relative_to(dumps_root).parts
            input_subset = rel_parts[1] if len(rel_parts) > 1 else None
            utterance_id = audio_path.stem

            # manifest.parquet: 発話ごとの音声メタ情報
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
            manifest_rows.append(manifest_row)

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
            index_rows.append(index_row)
        except Exception as e:
            print(f"Error processing {meta_path}: {e}")
    manifest_df = pl.DataFrame(manifest_rows)
    manifest_df.write_parquet(manifest_path)
    index_df = pl.DataFrame(index_rows)
    index_df.write_parquet(index_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create manifest.parquet and index.parquet from meta.json files under dumps root.")
    parser.add_argument("dumps_root", type=str, help="Path to dumps directory (e.g. dumps)")
    parser.add_argument("manifest_path", type=str, help="Output manifest Parquet file path")
    parser.add_argument("index_path", type=str, help="Output index Parquet file path")
    args = parser.parse_args()
    main(args.dumps_root, args.manifest_path, args.index_path)
