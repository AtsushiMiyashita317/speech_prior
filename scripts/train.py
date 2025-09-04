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
    from prior.nn.functional import (
        series_covariance, 
        series_covariance_mask, 
        series_correlation, 
        series_variance,
        kld_gaussian
    )
    from prior.nn.model.cnn1d import CNN1dKernel
    from prior.nn.model.feature_extractor import BothsidePaddedWav2Vec2Model

    def forward_one_step(
        batch, model, prototype, device, kernel_bins
    ):
        wave, teacher, wave_mask, teacher_mask = batch
        teacher_mask = teacher_mask.to(device)
        b = wave.size(0)
        n = kernel_bins // b
        cov_teacher = series_covariance(teacher.to(device), n)
        feature = model.module.forward(
            wave.to(device), 
            attention_mask=wave_mask.to(device)
        ).last_hidden_state
        feature = feature * teacher_mask.unsqueeze(-1)
        cov_mask = series_covariance_mask(teacher_mask, n)
        cov_feature = series_covariance(feature, n)
        cov_feature = cov_feature * cov_mask
        cov_feature = torch.stack([cov_feature, cov_feature], dim=-1)
        cov_feature = prototype.module.forward(cov_feature, mask=cov_mask).select(-1, 1)
        
        v = series_variance(cov_teacher)
        stable_mask = series_covariance_mask(v.lt(2.0).long(), n)
        cov_mask = cov_mask.logical_and(stable_mask)
        cov_teacher = cov_teacher * cov_mask
        cov_feature = cov_feature * cov_mask
        
        cor_feature = series_correlation(cov_feature)
        cor_teacher = series_correlation(cov_teacher)
        
        
        x = cor_feature.masked_select(cov_mask)
        y = cor_teacher.masked_select(cov_mask)
        cor_loss = torch.nn.functional.mse_loss(x, y)
        # loss = torch.nn.functional.mse_loss(x, y)
        vx = series_variance(cov_feature).masked_select(teacher_mask.bool())
        vy = series_variance(cov_teacher).masked_select(teacher_mask.bool())
        var_loss = kld_gaussian(vy, vx).mean()
        return cor_feature, cor_teacher, cor_loss, var_loss


    def train_one_epoch(
        train_dataloader,
        model, prototype, 
        optimizer, scheduler,
        device, rank, pbar, kernel_bins
    ):
        for batch_idx, batch in enumerate(train_dataloader):
            _, _, cor_loss, var_loss = forward_one_step(batch, model, prototype, device, kernel_bins)
            loss = cor_loss + var_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if rank == 0:
                wandb.log({
                    "train/cor_loss": cor_loss.item(),
                    "train/var_loss": var_loss.item(),
                    "train/loss": loss.item(),
                })
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=loss.item(),
                    )

        if rank == 0 and pbar:
            pbar.reset()

    @torch.no_grad()
    def plot(batch, model, prototype, device, kernel_bins):
        x, y, _, _ = forward_one_step(batch, model, prototype, device, kernel_bins)

        x = x.flatten(0, 2)
        y = y.flatten(0, 2)

        min_val = min(x.min().item(), y.min().item())
        max_val = max(x.max().item(), y.max().item())
        # min_val = 0
        # max_val = 3.5

        fig, ax = plt.subplots(1, 2, figsize=(40, 24))
        lx = ax[0].imshow(x.cpu(), vmin=min_val, vmax=max_val, aspect='auto', interpolation='nearest')
        ax[0].set_title("Predicted")
        ly = ax[1].imshow(y.cpu(), vmin=min_val, vmax=max_val, aspect='auto', interpolation='nearest')
        ax[1].set_title("Target")
        fig.colorbar(lx, ax=ax[0])
        fig.colorbar(ly, ax=ax[1])

        plt.tight_layout()

        # wandb 画像記録
        wandb.log({"plot": wandb.Image(fig)})
        plt.close(fig)

    @torch.no_grad()
    def validate(
        valid_dataloader,
        model, prototype, 
        device, rank, pbar, world_size, kernel_bins
    ):
        cor_loss_list = []
        var_loss_list = []
        # loss_list = []
        first_batch = None
        for batch_idx, batch in enumerate(valid_dataloader):
            if rank == 0 and first_batch is None:
                first_batch = batch
            _, _, cor_loss, var_loss = forward_one_step(batch, model, prototype, device, kernel_bins)
            # _, _, loss = forward_one_step(batch, model, prototype, device, kernel_bins)
            
            # Gather losses from all processes
            cor_loss_tensor = torch.tensor([cor_loss.item()], device=device)
            gathered_cor_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(gathered_cor_losses, cor_loss_tensor)
            
            var_loss_tensor = torch.tensor([var_loss.item()], device=device)
            gathered_var_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(gathered_var_losses, var_loss_tensor)
            
            # loss_tensor = torch.tensor([loss.item()], device=device)
            # gathered_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
            # dist.all_gather(gathered_losses, loss_tensor)

            if rank == 0:
                # Extend list with losses from all processes
                for l in gathered_cor_losses:
                    cor_loss_list.append(l.item())
                for l in gathered_var_losses:
                    var_loss_list.append(l.item())
                # for l in gathered_losses:
                #     loss_list.append(l.item())
                if pbar:
                    pbar.update(1)

        if rank == 0 and pbar:
            pbar.reset()

        cor_loss_avg = sum(cor_loss_list) / len(cor_loss_list) if cor_loss_list else 0.0
        var_loss_avg = sum(var_loss_list) / len(var_loss_list) if var_loss_list else 0.0
        # loss_avg = sum(loss_list) / len(loss_list) if loss_list else 0.0
        if rank == 0:
            wandb.log({
                "valid/cor_loss": cor_loss_avg, 
                "valid/var_loss": var_loss_avg,
                "valid/loss": cor_loss_avg + var_loss_avg
                # "valid/loss": loss_avg
            })
        return cor_loss_avg + var_loss_avg, first_batch
    
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

    # チェックポイント復帰
    start_epoch = 0
    if args.resume_checkpoint is not None and os.path.isfile(args.resume_checkpoint):
        map_location = device if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(args.resume_checkpoint, map_location=map_location)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        prototype.module.load_state_dict(checkpoint['prototype_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"[rank {rank}] Resumed from checkpoint: {args.resume_checkpoint} (epoch {start_epoch})")

    if rank == 0:
        print("\nオプティマイザとスケジューラを設定しました。")
        print("Optimizer:", optimizer.__class__.__name__)
        print("Scheduler:", scheduler.__class__.__name__)

    train_pbar = tqdm(total=len(train_dataloader)) if rank == 0 else None
    valid_pbar = tqdm(total=len(valid_dataloader)) if rank == 0 else None
    for epoch in range(start_epoch, num_epochs):
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
            args.kernel_bins,
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
            kernel_bins=args.kernel_bins
        )
        if rank == 0 and plot_batch is not None:
            plot(plot_batch, model, prototype, device, args.kernel_bins)
        

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
    parser.add_argument("--batch_bins", type=int, default=6000)
    parser.add_argument("--num_folds", type=int, default=256)
    parser.add_argument("--hop_length", type=int, default=320)
    parser.add_argument("--kernel_bins", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")

    args = parser.parse_args()

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if world_size == 0 and torch.cuda.is_available():
        print("Warning: CUDA is available, but world_size is 0. Did you mean to use CPUs?")
        world_size = 1 # Fallback to 1 process on CPU

    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

