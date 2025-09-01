import os
import h5py
import polars as pl
import torch
import torchaudio

from prior.utils.data import DiskLRU

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
        
        self.feature_range = {}
        for row in self.features_df.iter_rows(named=True):
            model = row["model"]
            layer = row["layer"]
            start = row["start"]
            end = row["end"]
            if model not in self.feature_range:
                self.feature_range[model] = {}
            self.feature_range[model][layer] = (start, end)

        self.D_total = 0
        for row in self.features_df.iter_rows(named=True):
            self.D_total += row["D"]
            
        self.utt_lens = [row["T"] for row in self.utterance_df.iter_rows(named=True)]

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        # raw_path = self.cache.ensure(self.raw_paths[utt_id])
        # wave, _ = self._load_waveform(raw_path)

        T = self.utt_lens[idx]
        h5_path = self.cache.ensure(self.hdf5_paths[utt_id])
        wave, features = self._load_hdf5(h5_path, T)

        return wave, features

    def _load_waveform(self, path: str) -> tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(path)  # [C, S]
        if wav.ndim == 2:
            if wav.shape[0] == 1:
                wav = wav[0]
            else:
                wav = wav.mean(dim=0)
        return wav, int(sr)

    def _load_hdf5(self, h5_path, T):
        with h5py.File(h5_path, "r") as f:
            group = f["utterance"]
            features = group["features"][:]
            waveform = group["waveform"][:]
            # if self.model_layers is None:
            #     features = dataset[:]
            # else:
            #     features = np.empty((T, self.D_total), order='C')
            #     # model_layers: {model: [layer, ...], ...}
            #     offset = 0
            #     for model, layers in self.model_layers.items():
            #         for layer in layers:
            #             start, end = self.feature_range[model][str(layer)]
            #             width = end - start
            #             dataset.read_direct(
            #                 features,
            #                 source_sel=np.s_[ :, start:end],
            #                 dest_sel=np.s_[ :, offset:offset+width])
            #             offset += width

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

