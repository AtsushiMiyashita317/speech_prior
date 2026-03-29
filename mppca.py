
def worker(proc_id, num_processes, dataset_info):
    print(f"Process {proc_id} started")
    from torch import save
    from prior.dataset.hdf5_dataset import HDF5Dataset
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import gc

    train_dataset = HDF5Dataset(
        dataset_info['base_dir'],
        dataset_info['hdf5_dir'],
        dataset_info['features_parquet'],
        dataset_info['utterance_parquet'],
        subset_list=dataset_info['subset_list'],
        stats_path=dataset_info['stats_path'],
    )
    print(f"Process {proc_id} loaded dataset")

    N = len(train_dataset)
    chunk_size = N // num_processes
    start_idx = proc_id * chunk_size
    end_idx = N if proc_id == num_processes - 1 else (proc_id + 1) * chunk_size
    indices = list(range(start_idx, end_idx))

    total_D = 0
    kernel_sum = None
    prefetch = 4  # Number of batches to prefetch
    
    def _load_batch(idx):
        batch = train_dataset[idx]
        return batch[1].cuda()
    
    with ThreadPoolExecutor(max_workers=prefetch) as executor:
        future_to_idx = {}
        idx_iter = iter(indices)
        # Prefetch initial batches
        for _ in range(prefetch):
            try:
                idx = next(idx_iter)
                future = executor.submit(_load_batch, idx)
                future_to_idx[future] = idx
            except StopIteration:
                break

        pbar = tqdm(total=len(indices), desc=f"proc{proc_id}")
        while future_to_idx:
            # As soon as a batch is loaded, process it
            done = next(as_completed(future_to_idx), None)
            if done is None:
                break
            idx = future_to_idx.pop(done)
            batch = done.result()
            features = batch
            D_batch = features.shape[0]
            total_D += D_batch
            gram_matrix = features.T @ features
            if kernel_sum is None:
                kernel_sum = gram_matrix
            else:
                kernel_sum += gram_matrix
            del features, gram_matrix, batch
            gc.collect()
            pbar.update(1)
            # Submit next batch if any left
            try:
                next_idx = next(idx_iter)
                future = executor.submit(_load_batch, next_idx)
                future_to_idx[future] = next_idx
            except StopIteration:
                pass
        pbar.close()

    out_path = f"dev_kernel_{proc_id}.pt"
    save({"kernel": kernel_sum.cpu(), "D": total_D}, out_path)

def main():
    from torch import save
    import torch.multiprocessing as mp

    base_dir = "/data/group1/z44542r/gitrepo/speech_prior"
    hdf5_dir = "/data/group1/z44542r/gitrepo/speech_prior/datasets"
    features_parquet = "features.parquet"
    utterance_parquet = "utterance.parquet"
    num_processes = 4
    
    # データセット情報をまとめる
    dataset_info = {
        'base_dir': base_dir,
        'hdf5_dir': hdf5_dir,
        'features_parquet': features_parquet,
        'utterance_parquet': utterance_parquet,
        'subset_list': ["train-clean-100", "train-clean-360", "train-other-500"],
        'stats_path': "/data/group1/z44542r/gitrepo/speech_prior/stats/train-clean-100.npz",
    }
    mp.spawn(worker, args=(num_processes, dataset_info), nprocs=num_processes, join=True)

    # 最後にメインプロセスで結果を集約
    import torch
    D = 0
    kernel = None
    for i in range(num_processes):
        result = torch.load(f"dev_kernel_{i}.pt")
        D += result["D"]
        if kernel is None:
            kernel = result["kernel"]
        else:
            kernel += result["kernel"]

    save({"kernel": kernel, "D": D}, "dev_kernel.pt")

if __name__ == "__main__":
    main()
