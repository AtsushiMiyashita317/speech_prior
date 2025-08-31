import os
import h5py
import polars as pl
import numpy as np
import threading
import shutil
from typing import Tuple
import torch
import torchaudio


def _async_copy(src, dst):
    def _copy():
        try:
            shutil.copy2(src, dst)
        except Exception:
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
    t = threading.Thread(target=_copy)
    t.start()

# ------------------------------------------------------------
# Disk LRU Cache (simple)
# ------------------------------------------------------------
class DiskLRU:
    """Very simple disk LRU cache for files.

    Files are identified by original absolute path; cached path is under cache_dir/aa/bb/<sha>.
    Eviction is by total size (approximate, bytes).
    """

    def __init__(self, base_dir: str, cache_dir: str, budget_bytes: int = 0) -> None:
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.budget = int(budget_bytes) if cache_dir else 0
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def ensure(self, relpath: str) -> str:
        """Ensure file is present in cache. Return cached absolute path (or original if disabled)."""
        abspath = os.path.join(self.base_dir, relpath)
        if not self.budget or not self.cache_dir:
            return abspath
        dst = os.path.join(self.cache_dir, relpath)
        if os.path.exists(dst):
            return dst

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        _async_copy(abspath, dst)

        return abspath


class HDF5Dataset:
    def __init__(
        self,
        base_dir,
        raw_dir,
        hdf5_dir,
        cache_dir,
        budget_bytes,
        features_parquet,
        utterance_parquet,
        model_layers=None
    ):
        # Parquetファイルの読み込み
        self.features_df = pl.read_parquet(os.path.join(hdf5_dir, features_parquet))
        self.utterance_df = pl.read_parquet(os.path.join(hdf5_dir, utterance_parquet))
        self.model_layers = model_layers
        self.cache = DiskLRU(base_dir, cache_dir, budget_bytes)

        self.raw_paths = {
            row["utt_id"]: os.path.join(raw_dir, row["path"] + ".flac")
            for row in self.utterance_df.iter_rows(named=True)
        }
        self.hdf5_paths = {
            row["utt_id"]: os.path.join(hdf5_dir, row["path"] + ".h5")
            for row in self.utterance_df.iter_rows(named=True)
        }
        self.utt_ids = list(self.hdf5_paths.keys())

        self.D_total = 0
        for row in self.features_df.iter_rows(named=True):
            self.D_total += row["D"]

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        raw_path = self.cache.ensure(self.raw_paths[utt_id])
        wave, _ = self._load_waveform(raw_path)


        h5_path = self.cache.ensure(self.hdf5_paths[utt_id])
        features = self._load_hdf5(h5_path)

        return wave, features

    def _load_waveform(self, path: str) -> Tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(path)  # [C, S]
        if wav.ndim == 2:
            if wav.shape[0] == 1:
                wav = wav[0]
            else:
                wav = wav.mean(dim=0)
        return wav.to(self.dtype), int(sr)

    def _load_hdf5(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            meta = dict(f["meta"].attrs)
            dataset = f["features"]
            T = dataset.shape[0]
            if self.model_layers is None:
                features = dataset[:]
            else:
                features = np.empty((T, self.D_total), order='C')
                # model_layers: {model: [layer, ...], ...}
                offset = 0
                for key, rng in meta.items():
                    model, layer = key.split('.', 1)
                    if model in self.model_layers and layer in self.model_layers[model]:
                        start, end = rng
                        width = end - start
                        dataset.read_direct(
                            features, 
                            src_sel=np.s_[ :, start:end],
                            dest_sel=np.s_[ :, offset:offset+width]
                        )
                        offset += width

        return torch.from_numpy(features)

    def get_feature_info(self, model, layer):
        # features.parquetからD次元数など取得
        df = self.features_df.filter(
            (pl.col("model") == model) & (pl.col("layer") == layer)
        )
        return df

    def get_utterance_info(self, utt_id):
        # utterance.parquetからTやパス取得
        df = self.utterance_df.filter(pl.col("utt_id") == utt_id)
        return df
