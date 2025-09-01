
import argparse
import numpy as np
from tqdm import tqdm

from src.prior.dataset.hdf5_dataset import HDF5Dataset

def main():
    parser = argparse.ArgumentParser(description="Calculate mean and variance of features in HDF5Dataset.")
    parser.add_argument('--base_dir', type=str, default='./', help='Base directory')
    parser.add_argument('--hdf5_dir', type=str, default='./datasets', help='HDF5 directory')
    parser.add_argument('--features_parquet', type=str, default='features.parquet', help='Features parquet filename')
    parser.add_argument('--utterance_parquet', type=str, default='utterance.parquet', help='Utterance parquet filename')
    parser.add_argument('--output', type=str, default='stats.npz', help='Output npz filename')
    args = parser.parse_args()

    # HDF5Datasetの初期化
    dataset = HDF5Dataset(
        args.base_dir,
        args.hdf5_dir,
        args.features_parquet,
        args.utterance_parquet
    )

    sum_ = None
    sum_sq = None
    total_count = 0

    # 走査して合計・二乗和・総数を集計
    for i in tqdm(range(len(dataset))):
        _, features = dataset[i]  # features: (T, D)
        features_np = features.numpy()
        if sum_ is None:
            sum_ = np.sum(features_np, axis=0)
            sum_sq = np.sum(features_np ** 2, axis=0)
        else:
            sum_ += np.sum(features_np, axis=0)
            sum_sq += np.sum(features_np ** 2, axis=0)
        total_count += features_np.shape[0]

    mean = sum_ / total_count
    var = sum_sq / total_count - mean ** 2

    # 保存
    np.savez(args.output, mean=mean, var=var)
    print(f"Saved {args.output}: mean shape {mean.shape}, var shape {var.shape}")

if __name__ == "__main__":
    main()
