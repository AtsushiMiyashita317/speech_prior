import os
import h5py
import polars as pl
import torch
import numpy as np

from prior.utils.data import DiskLRU

class HDF5Dataset:
    def __init__(
        self,
        base_dir,
        hdf5_dir,
        features_parquet,
        utterance_parquet,
        cache_dir=None,
        budget_bytes=0,
        model_layers=None,
        subset_list=None,
        stats_path=None,
    ):
        # Parquetファイルの読み込み
        self.features_df = pl.read_parquet(os.path.join(hdf5_dir, features_parquet))
        self.utterance_df = pl.read_parquet(os.path.join(hdf5_dir, utterance_parquet))
        self.model_layers = model_layers
        self.cache = DiskLRU(base_dir, cache_dir, budget_bytes)
        
        if subset_list is not None:
            self.utterance_df = self.utterance_df.filter(pl.col("subset").is_in(subset_list))

        self.hdf5_paths = {
            row["utt_id"]: os.path.join(hdf5_dir, row["path"] + ".h5")
            for row in self.utterance_df.iter_rows(named=True)
        }
        self.utt_ids = list(self.hdf5_paths.keys())
        
        if stats_path is not None:
            stats = np.load(stats_path)
            mean = torch.tensor(stats["mean"])
            smean = torch.tensor(stats["smean"])
        else:
            mean = None
            smean = None
        
        self.feature_range = {}
        for row in self.features_df.iter_rows(named=True):
            model = row["model"]
            layer = row["layer"]
            start = row["start"]
            end = row["end"]
            if model not in self.feature_range:
                self.feature_range[model] = {}
            self.feature_range[model][layer] = (start, end)
            
            # if mean is not None:
            #     mean[start:end] = mean[start:end].mean()
            #     smean[start:end] = smean[start:end].mean()
                
        if mean is not None:
            self.mean = mean
            std = torch.sqrt(smean - mean**2)
            self.std = std
        else:
            self.mean = None
            self.std = None

        self.D_total = 0
        for row in self.features_df.iter_rows(named=True):
            self.D_total += row["D"]
            
        self.utt_lens = [row["T"] for row in self.utterance_df.iter_rows(named=True)]

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]

        T = self.utt_lens[idx]
        h5_path = self.cache.ensure(self.hdf5_paths[utt_id])
        wave, features = self._load_hdf5(h5_path, T)
        wave = (wave - wave.mean()) / (wave.std() + 1e-7)
        
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / (self.std + 1e-7)

        return wave, features

    def _load_hdf5(self, h5_path, T):
        with h5py.File(h5_path, "r") as f:
            group = f["utterance"]
            features = group["features"][:]
            waveform = group["waveform"][:]
            # TODO subset access

        return torch.from_numpy(waveform), torch.from_numpy(features)

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

