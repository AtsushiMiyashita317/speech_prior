import argparse
import os
import torch.multiprocessing as mp


def main_worker(rank, world_size, args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from tqdm import tqdm
    from transformers import Wav2Vec2Config, Wav2Vec2Model, get_linear_schedule_with_warmup
    from matplotlib import pyplot as plt
    import wandb

    from prior.dataset.hdf5_dataset import HDF5Dataset
    from prior.utils.data import SeriesCollator, FoldedLengthBatchSampler
    from prior.nn.functional import series_covariance, series_covariance_mask, series_correlation
    from prior.nn.model.cnn1d import CNN1dKernel
    from prior.nn.model.feature_extractor import BothsidePaddedWav2Vec2Model

    def forward_one_step(
        batch, model, prototype, device
    ):
        wave, teacher, wave_mask, teacher_mask = batch
        b = wave.size(0)
        n = 256 // b
        cov_teacher = series_covariance(teacher.to(device), n)
        feature = model.module.forward(
            wave.to(device), 
            attention_mask=wave_mask.to(device)
        ).last_hidden_state
        feature = feature * teacher_mask.to(device).unsqueeze(-1)
        cov_mask = series_covariance_mask(teacher_mask.to(device), n)
        cov_feature = series_covariance(feature, n)
        cov_feature = cov_feature * cov_mask
        cov_feature = torch.stack([cov_feature, cov_feature], dim=-1)
        cov_feature = prototype.module.forward(cov_feature).select(-1, 1)
        cor_feature = series_correlation(cov_feature)
        cor_teacher = series_correlation(cov_teacher)
        x = cor_feature.masked_select(cov_mask)
        y = cor_teacher.masked_select(cov_mask)
        loss = torch.nn.functional.mse_loss(x, y)
        return x, y, cor_feature, cor_teacher, loss


    def train_one_epoch(
        train_dataloader,
        model, prototype, 
        optimizer, scheduler,
        device, rank, pbar,
    ):
        for batch_idx, batch in enumerate(train_dataloader):
            x, y, _, _, loss = forward_one_step(batch, model, prototype, device)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if rank == 0:
                wandb.log({"train/loss": loss.item()})
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=loss.item(),
                        x_mean=x.mean().item(),
                        x_std=x.std().item(),
                        y_mean=y.mean().item(),
                        y_std=y.std().item(),
                    )

        if rank == 0 and pbar:
            pbar.reset()

    @torch.no_grad()
    def plot(batch, model, prototype, device, epoch=None):
        _, _, x, y, _ = forward_one_step(batch, model, prototype, device)

        x = x.flatten(0, 2)
        y = y.flatten(0, 2)

        # min_val = min(x.min().item(), y.min().item())
        # max_val = max(x.max().item(), y.max().item())
        min_val = -1
        max_val = 1

        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        lx = ax[0].imshow(x.cpu(), vmin=min_val, vmax=max_val, aspect='auto')
        ax[0].set_title("Predicted")
        ly = ax[1].imshow(y.cpu(), vmin=min_val, vmax=max_val, aspect='auto')
        ax[1].set_title("Target")
        fig.colorbar(lx, ax=ax[0])
        fig.colorbar(ly, ax=ax[1])

        plt.tight_layout()

        # wandb 画像記録
        if epoch is not None:
            wandb.log({"plot": wandb.Image(fig)})
        plt.close(fig)

    @torch.no_grad()
    def validate(
        valid_dataloader,
        model, prototype, 
        device, rank, pbar, world_size,
    ):
        loss_list = []
        first_batch = None
        for batch_idx, batch in enumerate(valid_dataloader):
            if rank == 0 and first_batch is None:
                first_batch = batch
            _, _, _, _, loss = forward_one_step(batch, model, prototype, device)
            
            # Gather losses from all processes
            loss_tensor = torch.tensor([loss.item()], device=device)
            gathered_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(gathered_losses, loss_tensor)

            if rank == 0:
                # Extend list with losses from all processes
                for l in gathered_losses:
                    loss_list.append(l.item())
                if pbar:
                    pbar.update(1)

        if rank == 0 and pbar:
            pbar.reset()

        loss_avg = sum(loss_list) / len(loss_list) if loss_list else 0.0
        if rank == 0:
            wandb.log({"valid/loss_avg": loss_avg})
        return loss_avg, first_batch
    
    print(f"[main_worker] rank={rank}, world_size={world_size}")
    # DDP setup
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)

    print(f"[main_worker] DDP initialized on rank={rank}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        wandb.init(project="speech_prior", config=vars(args))
        # wandbの実行ディレクトリ内にチェックポイント用のフォルダを作成
        checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = None

    train_dataset = HDF5Dataset(
        args.base_dir,
        args.hdf5_dir,
        args.features_parquet,
        args.utterance_parquet,
        cache_dir=args.cache_dir,
        budget_bytes=6 << 40,
        subset_list=["train-clean-100"],
        stats_path="stats/train-clean-100.npz",
    )

    valid_dataset = HDF5Dataset(
        args.base_dir,
        args.hdf5_dir,
        args.features_parquet,
        args.utterance_parquet,
        cache_dir=args.cache_dir,
        budget_bytes=6 << 40,
        subset_list=["dev-clean"],
        stats_path="stats/dev-clean.npz",
    )

    train_sampler = FoldedLengthBatchSampler(
        train_dataset.utt_lens,
        args.batch_bins,
        args.num_folds,
        shuffle=True,
        num_replicas=world_size,
        rank=rank
    )

    valid_sampler = FoldedLengthBatchSampler(
        valid_dataset.utt_lens,
        args.batch_bins,
        args.num_folds,
        shuffle=False,
        num_replicas=world_size,
        rank=rank
    )

    train_collator = SeriesCollator(args.hop_length, pad_bothside=True)
    valid_collator = SeriesCollator(args.hop_length, pad_bothside=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collator.collate_batch,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        collate_fn=valid_collator.collate_batch,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
    model = BothsidePaddedWav2Vec2Model(config)
    model.train()
    prototype = CNN1dKernel(num_layers=3, kernel_size=5)

    model.to(device)
    prototype.to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    prototype = DDP(prototype, device_ids=[rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(prototype.parameters()), lr=1e-4, weight_decay=0.01)

    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    if rank == 0:
        print("\nオプティマイザとスケジューラを設定しました。")
        print("Optimizer:", optimizer.__class__.__name__)
        print("Scheduler:", scheduler.__class__.__name__)

    train_pbar = tqdm(total=len(train_dataloader)) if rank == 0 else None
    valid_pbar = tqdm(total=len(valid_dataloader)) if rank == 0 else None
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        prototype.train()
        
        train_one_epoch(
            train_dataloader,
            model,
            prototype,
            optimizer,
            scheduler,
            device,
            rank,
            train_pbar,
        )

        model.eval()
        prototype.eval()
        loss_avg, plot_batch = validate(
            valid_dataloader,
            model,
            prototype,
            device,
            rank=rank,
            pbar=valid_pbar,
            world_size=world_size,
        )
        if rank == 0 and plot_batch is not None:
            plot(plot_batch, model, prototype, device, epoch)
        

        if epoch % 10 == 0 and rank == 0:
            # チェックポイントの保存 (rank 0 のみ)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'prototype_state_dict': prototype.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_avg,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
                
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


def main():
    import torch
    parser = argparse.ArgumentParser(description="Train a Wav2Vec2 model with custom dataset")
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument("--hdf5_dir", type=str, default="./datasets")
    parser.add_argument("--features_parquet", type=str, default="features.parquet")
    parser.add_argument("--utterance_parquet", type=str, default="utterance.parquet")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_bins", type=int, default=8000)
    parser.add_argument("--num_folds", type=int, default=64)
    parser.add_argument("--hop_length", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")

    args = parser.parse_args()

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if world_size == 0 and torch.cuda.is_available():
        print("Warning: CUDA is available, but world_size is 0. Did you mean to use CPUs?")
        world_size = 1 # Fallback to 1 process on CPU

    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

