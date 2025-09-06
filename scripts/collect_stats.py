
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from prior.dataset.hdf5_dataset import HDF5Dataset

def main():
    parser = argparse.ArgumentParser(description="Calculate mean and variance of features in HDF5Dataset.")
    parser.add_argument('--base_dir', type=str, default='./', help='Base directory')
    parser.add_argument('--hdf5_dir', type=str, default='./datasets', help='HDF5 directory')
    parser.add_argument('--subset', type=str, default=None, help='Subset list to filter utterances')
    parser.add_argument('--features_parquet', type=str, default='features.parquet', help='Features parquet filename')
    parser.add_argument('--utterance_parquet', type=str, default='utterance.parquet', help='Utterance parquet filename')
    parser.add_argument('--output', type=str, default='stats.npz', help='Output npz filename')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use')
    args = parser.parse_args()

    # HDF5Datasetの初期化
    dataset = HDF5Dataset(
        args.base_dir,
        args.hdf5_dir,
        args.features_parquet,
        args.utterance_parquet,
        subset_list=[args.subset] if args.subset is not None else None,
    )
    
    
    def wrapper(i):
        _, features = dataset[i]  # features: (T, D)
        features_np = features.numpy().astype(np.float64)
        sum_ = np.sum(features_np, axis=0)
        sum_sq = np.sum(features_np ** 2, axis=0)
        count = features_np.shape[0]
        
        return sum_, sum_sq, count
    
    if args.num_samples is not None:
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
    else:
        indices = range(len(dataset))
        
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(wrapper, indices), total=len(indices)))
        
    sum_ = np.stack([r[0] for r in results], axis=0).sum(axis=0)
    sum_sq = np.stack([r[1] for r in results], axis=0).sum(axis=0)
    count = sum([r[2] for r in results])
    
    mean = sum_ / count
    smean = sum_sq / count

    # 保存
    np.savez(args.output, mean=mean, smean=smean)
    print(f"Saved {args.output}: mean shape {smean.shape}, smean shape {smean.shape}")

if __name__ == "__main__":
    main()
