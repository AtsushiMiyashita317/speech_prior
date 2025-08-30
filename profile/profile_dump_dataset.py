import argparse
import time
from tqdm import tqdm
import torch
from prior.dataset.dump_dataset import DumpDataset, InitAndPerWorkerShuffleSampler, collate_nested_batch

# テスト用のdumpsディレクトリ（実データが格納されているパスに合わせて変更してください）
DUMPS_ROOT = "./"
MANIFEST = "dumps/manifest.parquet"
INDEX = "dumps/index.parquet"

def profile_dump_dataset_basic(cache_dir):
    base_dir = DUMPS_ROOT
    input_subset = 'dev-clean'
    ds = DumpDataset(
        base_dir=base_dir,
        index_parquet=INDEX,
        manifest_parquet=MANIFEST,
        layers_by_model={
            "facebook/hubert-base-ls960": ["layer6", "layer9", "layer11"],
            "facebook/wav2vec2-base": ["layer6", "layer9", "layer11"],
        },
        input_subsets=input_subset,
        return_waveform=True,
        cache_dir=cache_dir,
        cache_bytes=0,
        cache_bytes_disk=5 << 40,
    )
    
    sampler = InitAndPerWorkerShuffleSampler(ds, seed=42)

    dataloader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=16,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_nested_batch,
        sampler=sampler,
    )

    for epoch in range(5):
        start = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            pbar.update(1)
        pbar.close()
        end = time.time()
        print(f"Time taken for one epoch {epoch}: {end - start:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile DumpDataset")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for caching dataset")
    args = parser.parse_args()
    profile_dump_dataset_basic(args.cache_dir)
