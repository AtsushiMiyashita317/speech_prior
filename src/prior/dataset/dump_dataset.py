from __future__ import annotations
import os
import random
import hashlib
import shutil
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
try:
    profile
except NameError:
    def profile(func): return func

import numpy as np
import polars as pl

import torchaudio
import torch


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _abspath(base_dir: str, rel: str) -> str:
    return os.path.abspath(os.path.join(base_dir, rel))

def _async_copy(src, dst):
    def _copy():
        try:
            shutil.copy2(src, dst)
        except Exception:
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
    t = threading.Thread(target=_copy)
    t.start()

@dataclass
class FeatureRef:
    path: str
    T: Optional[int]
    D: int
    fps: int
    arch: str
    frontend: Optional[str] = None


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

    @profile
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

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class DumpDataset(torch.utils.data.Dataset):
    """Dataset for dumped features + original audio, per spec v0.3.

    Key behaviors implemented:
      - Two-level catalog: manifest.parquet (utterance) + index.parquet (feature files)
      - layers_by_model to specify multi-model/multi-layer selection
      - input_subsets list to concatenate subsets
      - Return *full* waveform + features dict in __getitem__ (no windowing)
      - SSD disk cache + memmap cache
    """

    @profile
    def __init__(
        self,
        base_dir: str,
        index_parquet: str,
        manifest_parquet: str,
        # selection / multi
        layers_by_model: Optional[Dict[str, List[str]]] = None,
        input_subsets: Optional["str | List[str]"] = None,
        select: Optional[Dict[str, Any]] = None,
        dedup: bool = True,
        # waveform
        return_waveform: bool = True,
        # normalization / join mode
        normalize: Optional[Dict[str, Any]] = None,
        feature_join: str = 'dict',
        project: str = 'pad',
        # I/O & caches
        dtype: torch.dtype = torch.float32,
        mmap: bool = True,
        cache_dir: Optional[str] = None,
        cache_bytes: int = 128 << 20,
        cache_bytes_disk: int = 0,
        # schema registry
        feature_schema_path: Optional[str] = None,
        schema_strict: bool = True,
        # verify / policy
        verify: Optional[Dict[str, Any]] = None,
        joinable: str = 'strict',
        on_error: str = 'skip+warn',
        # distributed
        shard: Optional[Tuple[int, int]] = None,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.base_dir = os.path.abspath(base_dir)
        self.index_parquet = os.path.join(self.base_dir, index_parquet)
        self.manifest_parquet = os.path.join(self.base_dir, manifest_parquet)
        self.layers_by_model = layers_by_model or {}
        self.input_subsets = ([input_subsets] if isinstance(input_subsets, str) else input_subsets) or []
        self.select = select or {}
        self.dedup = dedup
        self.return_waveform = return_waveform
        self.normalize = normalize or {"kind": "none"}
        self.feature_join = feature_join
        self.project = project
        self.dtype = dtype
        self.mmap = mmap
        self.disk_cache = DiskLRU(base_dir, cache_dir, cache_bytes_disk)
        self.mem_cache_budget = int(cache_bytes)
        self.mem_cache: Dict[str, np.ndarray] = {}
        self.mem_cache_order: List[str] = []
        self.feature_schema_path = feature_schema_path
        self.schema_strict = schema_strict
        self.verify = verify or {}
        self.joinable = joinable
        self.on_error = on_error
        self.shard = shard
        self.drop_last = drop_last
        self.seed = seed

        # Load catalogs, build join map and index list
        self._load_catalogs()
        self._build_join_map()
        self._materialize_utterance_list()

    # ------------------------- catalog & join map -------------------------
    def _read_parquet(self, path: str):
        return pl.read_parquet(path)

    def _load_catalogs(self) -> None:
        if not os.path.exists(self.index_parquet):
            raise FileNotFoundError(f"index.parquet not found: {self.index_parquet}")
        if not os.path.exists(self.manifest_parquet):
            raise FileNotFoundError(f"manifest.parquet not found: {self.manifest_parquet}")
        df_index = self._read_parquet(self.index_parquet)
        self.df_manifest = self._read_parquet(self.manifest_parquet)

        # Filter manifest by input_subsets
        if self.input_subsets:
            self.df_manifest = self.df_manifest.filter(pl.col('input_subset').is_in(self.input_subsets))
        # dedup by utterance_id
        if self.dedup:
            self.df_manifest = self.df_manifest.unique('utterance_id', keep='first')

        # Semi-join index by available utterance_ids
        ids = self.df_manifest.select('utterance_id')
        df_index = df_index.join(ids, on='utterance_id', how='inner')

        # Apply select filters (simple AND where possible)
        for k, v in (self.select or {}).items():
            df_index = df_index.filter(pl.col(k) == v)

        self.df_wav = df_index.filter(pl.col('arch') == 'raw')

        self.df_feature = df_index

        # Filter by layers_by_model
        if self.layers_by_model:
            # allowed: list of dicts [{'model_id': ..., 'layer': ...}, ...]
            allowed = [
                {'model_id': mid, 'layer': ly}
                for mid, layers in self.layers_by_model.items()
                for ly in layers
            ]
            mid_ok = pl.col('model_id').is_in([d['model_id'] for d in allowed])
            ly_ok = pl.struct(['model_id', 'layer']).is_in(allowed)
            self.df_feature = self.df_feature.filter(mid_ok & ly_ok)

        # Sort for locality: utterance_id -> model_id -> layer
        self.df_feature = self.df_feature.sort(['utterance_id', 'model_id', 'layer'])

        # Load feature schema registry (optional)
        self.schema: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if self.feature_schema_path and os.path.exists(self.feature_schema_path):
            df = self._read_parquet(self.feature_schema_path)
            for row in df.iter_rows(named=True):
                key = (row['model_id'], row['layer'])
                self.schema[key] = dict(row)

    def _build_join_map(self) -> None:
        # Manifest map: utterance_id -> audio info
        self.utt_map: Dict[str, Dict[str, Any]] = {
            row['utterance_id']: row for row in self.df_manifest.iter_rows(named=True)
        }

        # Join map: utterance_id -> model_id -> layer -> FeatureRef
        self.join_map: Dict[str, Dict[str, Dict[str, FeatureRef]]] = {}
        for row in self.df_feature.iter_rows(named=True):
            uid = row['utterance_id']
            mid = row['model_id']
            layer = row['layer']
            schema_row = self.schema.get((mid, layer), {})
            ref = FeatureRef(
                path=row['dump_path_rel'],
                T=row.get('T'),
                D=row.get('D', None),
                fps=int(row['fps']) if row.get('fps') is not None else int(schema_row.get('fps', 0)),
                arch=row.get('arch', schema_row.get('arch', '')),
                frontend=row.get('frontend', schema_row.get('frontend')),
            )
            self.join_map.setdefault(uid, {}).setdefault(mid, {})[layer] = ref

        self.wav_map: Dict[str, FeatureRef] = {}
        for row in self.df_wav.iter_rows(named=True):
            uid = row['utterance_id']
            mid = row['model_id']
            layer = row['layer']
            schema_row = self.schema.get((mid, layer), {})
            ref = FeatureRef(
                path=row['dump_path_rel'],
                T=row.get('T'),
                D=row.get('D', None),
                fps=int(row['fps']) if row.get('fps') is not None else int(schema_row.get('fps', 0)),
                arch=row.get('arch', schema_row.get('arch', '')),
                frontend=row.get('frontend', schema_row.get('frontend')),
            )
            self.wav_map[uid] = ref

    def _materialize_utterance_list(self) -> None:
        required_pairs = [(mid, ly) for mid, lys in self.layers_by_model.items() for ly in lys]
        uids = []
        for uid, models in self.join_map.items():
            ok = True
            if required_pairs and self.joinable == 'strict':
                for mid, ly in required_pairs:
                    if mid not in models or ly not in models[mid]:
                        ok = False
                        break
            if ok:
                uids.append(uid)
        uids.sort()

        if self.shard is not None:
            rank, world = self.shard
            if world <= 0:
                raise ValueError('world must be > 0')
            shard_uids = uids[rank::world]
            if self.drop_last:
                max_len = (len(uids) // world) * 1
                shard_uids = uids[rank*max_len:(rank+1)*max_len]
            self.uids = shard_uids
        else:
            self.uids = uids

    # ------------------------- dataset protocol -------------------------
    def __len__(self) -> int:
        return len(self.uids)

    def _load_waveform(self, audio_rel: str, sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        cached = self.disk_cache.ensure(audio_rel)
        wav, sr = torchaudio.load(cached)  # [C, S]
        if wav.ndim == 2:
            if wav.shape[0] == 1:
                wav = wav[0]
            else:
                wav = wav.mean(dim=0)
        return wav.to(self.dtype), int(sr)

    @profile
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid = self.uids[idx]
        utt = self.utt_map[uid]
        audio_rel_path = utt['audio_path_rel']

        # waveform (full)
        wav_tensor: Optional[torch.Tensor] = None
        sr = int(utt['sample_rate']) if 'sample_rate' in utt and utt['sample_rate'] is not None else None
        if self.return_waveform:
            audio_rel = self.wav_map[uid].path
            wav_tensor, sr_loaded = self._load_waveform(audio_rel, sample_rate=sr)
            if sr is None:
                sr = sr_loaded
        S = int(wav_tensor.numel()) if wav_tensor is not None else int(utt['num_samples'])
        audio_dict = {
            'waveform': wav_tensor if wav_tensor is not None else None,
            'sample_rate': int(sr or 16000),
            'lengths': S,
        }

        # features (full)
        features: Dict[str, Dict[str, Dict[str, Any]]] = {}
        lengths_frames: Dict[str, int] = {}
        models = self.join_map[uid]
        for mid, layers in self.layers_by_model.items():
            if mid not in models:
                if self.joinable == 'strict':
                    raise KeyError(f"missing model {mid} for utterance {uid}")
                else:
                    continue
            features[mid] = {}
            for ly in layers:
                ref = models[mid].get(ly)
                if ref is None:
                    if self.joinable == 'strict':
                        raise KeyError(f"missing layer {mid}:{ly} for utterance {uid}")
                    else:
                        continue
                npy = self._load_feature(ref.path)
                arr = torch.tensor(npy, dtype=self.dtype)  # [T, D]
                features[mid][ly] = {
                    'feature': arr,
                    'lengths': arr.shape[0],
                    'fps': ref.fps,
                }
                lengths_frames[mid] = arr.shape[0]

        meta = {
            'sample_id': uid,
            'utterance_id': uid,
            'audio_path_rel': audio_rel_path,
            'audio_sha256': utt.get('audio_sha256'),
            'sample_rate': int(sr or 16000),
            'num_samples': int(utt.get('num_samples', S)),
            'duration_sec': float(utt.get('duration_sec', S / float(sr or 16000))),
        }

        out = {
            'audio': audio_dict,
            'features': features,
            'meta': meta,
        }
        return out

    # -- feature loading with memmap + disk cache
    @profile
    def _load_feature(self, path_rel: str) -> np.ndarray:
        import line_profiler
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        profiler = line_profiler.LineProfiler()
        
        def tmp():
            key = path_rel
            if key in self.mem_cache:
                try:
                    self.mem_cache_order.remove(key)
                except ValueError:
                    pass
                self.mem_cache_order.append(key)
                return self.mem_cache[key]
            cached = self.disk_cache.ensure(path_rel)
            
            arr = np.load(cached, mmap_mode='r' if self.mmap else None)
            if self.mem_cache_budget > 0:
                self.mem_cache[key] = arr
                self.mem_cache_order.append(key)
                used = sum(self.mem_cache[k].nbytes for k in self.mem_cache)
                while used > self.mem_cache_budget and self.mem_cache_order:
                    old = self.mem_cache_order.pop(0)
                    used -= self.mem_cache[old].nbytes
                    del self.mem_cache[old]
            return arr
        
        result = profiler(tmp)()
        # 保存
        profiler.dump_stats(f"lineprofile_worker{worker_id}.lprof")
        return result
        


# ------------------------------------------------------------
# Sampler
# ------------------------------------------------------------

class InitAndPerWorkerShuffleSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, shuffle=True, seed=0):
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed
        # 初期化時に一度シャッフル
        indices = list(range(self.num_samples))
        if shuffle:
            random.seed(seed)
            random.shuffle(indices)
        self.base_indices = indices
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_indices = self.base_indices
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            worker_indices = self.base_indices[worker_id::num_workers]
        # 各ワーカー範囲内で毎エポックシャッフル
        if self.shuffle:
            random.seed(self.seed + self.epoch + (worker_info.id if worker_info else 0))
            worker_indices = worker_indices.copy()
            random.shuffle(worker_indices)
        return iter(worker_indices)

    def __len__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.num_samples
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            return len(self.base_indices[worker_id::num_workers])

# ------------------------------------------------------------
# Collate
# ------------------------------------------------------------

@profile
def collate_nested_batch(batch: List[Dict[str, Any]], pad_value: float = 0.0) -> Dict[str, Any]:
    """Pad/stack nested DumpDataset samples into a batch dict.

    - audio.waveform -> [B, S_max]
    - features[model][layer] -> [B, T_max, D]
    """

    B = len(batch)
    # audio
    s_lens = [b['audio']['lengths'] for b in batch]
    Smax = max(s_lens) if s_lens else 0
    batch_wave = torch.zeros((B, Smax), dtype=torch.float32)
    for i, b in enumerate(batch):
        wf = b['audio']['waveform']
        batch_wave[i, :wf.shape[0]].copy_(wf)

    # features
    features_out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    frames_out: Dict[str, List[int]] = {}
    all_models = set()
    all_layers: Dict[str, set] = {}
    for b in batch:
        for mid, L in b['features'].items():
            all_models.add(mid)
            all_layers.setdefault(mid, set()).update(L.keys())
    for mid in sorted(all_models):
        features_out[mid] = {}
        for ly in sorted(all_layers[mid]):
            seqs = []
            lens = []
            fps = None
            for b in batch:
                x_dict = b['features'].get(mid, {}).get(ly)
                if x_dict is not None:
                    x = x_dict['feature']
                    D = x.shape[1]
                    lens.append(x.shape[0])
                    seqs.append(x)
                    if fps is None:
                        fps = x_dict.get('fps', None)
            Tmax = max(lens) if lens else 0
            feature = torch.zeros((len(seqs), Tmax, D), dtype=torch.float32)
            for i, x in enumerate(seqs):
                feature[i, :x.shape[0], :].copy_(x)
                
            features_out[mid][ly] = {
                'feature': feature,
                'lengths': torch.tensor(lens, dtype=torch.long),
                'fps': fps,
            }
            frames_out[mid] = lens

    meta_list = [b['meta'] for b in batch]

    return {
        'audio': {
            'waveform': batch_wave,
            'sample_rate': batch[0]['audio']['sample_rate'] if B > 0 else None,
            'lengths': torch.tensor(s_lens, dtype=torch.long) if s_lens else None,
        },
        'features': features_out,
        'meta_list': meta_list,
    }
