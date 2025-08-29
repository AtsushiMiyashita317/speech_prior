import os
import pytest
import torch
from prior.dataset.dump_dataset import DumpDataset

# テスト用のdumpsディレクトリ（実データが格納されているパスに合わせて変更してください）
DUMPS_ROOT = "./"
MANIFEST = "dumps/manifest.parquet"
INDEX = "dumps/index.parquet"

@pytest.mark.parametrize("input_subset", [None, "dev-clean"])
def test_dump_dataset_basic(input_subset):
    base_dir = DUMPS_ROOT
    ds = DumpDataset(
        base_dir=base_dir,
        index_parquet=INDEX,
        manifest_parquet=MANIFEST,
        layers_by_model={"speechbrain/spkrec-xvect-voxceleb": ["blocks.0"]},
        input_subsets=input_subset,
        return_waveform=True,
        cache_dir=None,
        cache_bytes=32 << 20,
        cache_bytes_disk=0,
    )
    assert len(ds) > 0, "Dataset should not be empty"
    sample = ds[0]
    assert "audio" in sample
    assert "features" in sample
    assert "meta" in sample
    # waveform shape
    wf = sample["audio"]["waveform"]
    assert isinstance(wf, torch.Tensor)
    assert wf.ndim == 1
    # features shape
    features = sample["features"]["speechbrain/spkrec-xvect-voxceleb"]["blocks.0"]["feature"]
    assert isinstance(features, torch.Tensor)
    assert features.ndim == 2
    # meta fields
    meta = sample["meta"]
    assert "utterance_id" in meta
    assert "audio_path_rel" in meta
    assert "sample_rate" in meta
    assert "num_samples" in meta
    assert "duration_sec" in meta

if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
