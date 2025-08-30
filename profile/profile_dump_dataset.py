import cProfile
import pstats
import torch
from prior.dataset.dump_dataset import DumpDataset, collate_nested_batch

# テスト用のdumpsディレクトリ（実データが格納されているパスに合わせて変更してください）
DUMPS_ROOT = "./"
MANIFEST = "dumps/manifest.parquet"
INDEX = "dumps/index.parquet"

def profile_dump_dataset_basic():
    base_dir = DUMPS_ROOT
    ds = DumpDataset(
        base_dir=base_dir,
        index_parquet=INDEX,
        manifest_parquet=MANIFEST,
        layers_by_model={"speechbrain/spkrec-xvect-voxceleb": ["blocks.0"]},
        input_subsets=['dev-clean'],
        return_waveform=True,
        cache_dir=None,
        cache_bytes=32 << 20,
        cache_bytes_disk=0,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_nested_batch,
    )

    batch = next(iter(dataloader))


if __name__ == "__main__":
    cProfile.run("profile_dump_dataset_basic()", 'profile_result')
    p = pstats.Stats('profile_result')
    p.sort_stats('cumtime').print_stats(20)
    