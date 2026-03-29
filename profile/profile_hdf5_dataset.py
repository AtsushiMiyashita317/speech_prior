import argparse
import time
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from prior.dataset.hdf5_dataset import HDF5Dataset, collate_batch
from prior.nn.functional import series_covariance
from prior.utils.data import RandomFoldedLengthBatchSampler

# テスト用のdumpsディレクトリ（実データが格納されているパスに合わせて変更してください）
BASE = "./"
RAW = "./data/librispeech/LibriSpeech"
HDF5 = "./datasets"
FEATURE = "features.parquet"
UTTERANCE = "utterance.parquet"

def profile_dump_dataset_basic(cache_dir):
    input_subset = 'dev-clean'
    ds = HDF5Dataset(
        base_dir=BASE,
        raw_dir=RAW,
        hdf5_dir=HDF5,
        features_parquet=FEATURE,
        utterance_parquet=UTTERANCE,
        # model_layers={
        #     "hubert": [6, 9, 11],
        #     "wav2vec": [6, 9, 11],
        # },
        cache_dir=cache_dir,
        budget_bytes=5 << 40,
    )
    
    # sampler = InitAndPerWorkerShuffleSampler(ds, seed=42)
    
    batch_sampler = RandomFoldedLengthBatchSampler(
        ds.utt_lens,
        batch_bins=4000,
        num_folds=16,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        num_workers=40,
        prefetch_factor=2,
        pin_memory=False,
        collate_fn=collate_batch,
        persistent_workers=True,
    )

    for epoch in range(5):
        start = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            batch_sampler.set_epoch(epoch)
            wave, features = batch
            features = features.cuda()
            cov = series_covariance(features, n=80//features.size(0))
            pbar.update(1)
        pbar.close()
        end = time.time()
        print(f"Time taken for one epoch {epoch}: {end - start:.4f} seconds")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile DumpDataset")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for caching dataset")
    args = parser.parse_args()
    profile_dump_dataset_basic(args.cache_dir)
