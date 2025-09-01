import argparse
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import Wav2Vec2Config, Wav2Vec2Model, get_linear_schedule_with_warmup
from matplotlib import pyplot as plt
import wandb

from prior.dataset.hdf5_dataset import HDF5Dataset
from prior.utils.data import SeriesCollator, FoldedLengthBatchSampler
from prior.nn.functional import series_covariance, series_covariance_mask
from prior.nn.model.cnn1d import CNN1dKernel


def forward_one_step(
    batch, model, prototype, device
):
    wave, teacher, wave_mask, teacher_mask = batch
    teacher = teacher.to(device)
    b = wave.size(0)
    n = 80 // b
    cov_teacher = series_covariance(teacher, n)
    wave = wave.to(device)
    wave_mask = wave_mask.to(device)
    feature = model.module.forward(wave, attention_mask=wave_mask).last_hidden_state
    cov_mask = series_covariance_mask(teacher_mask.to(device), n)
    cov_feature = series_covariance(feature, n)
    cov_feature = cov_feature * cov_mask
    kernel = torch.stack([cov_feature, cov_feature], dim=-1)
    ntk = prototype.module.forward(kernel).select(-1, 1)
    x = ntk.masked_select(cov_mask)
    y = cov_teacher.masked_select(cov_mask) + 1.0
    loss = torch.nn.functional.mse_loss(x, y)
    return x, y, ntk, cov_teacher, loss


def train_one_epoch(
    train_dataloader,
    model, prototype,
    optimizer,
    scheduler,
    device,
    rank,
    pbar,
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
                )

    if rank == 0 and pbar:
        pbar.reset()

@torch.no_grad()
def plot(batch, model, prototype, device, epoch=None):
    _, _, x, y, _ = forward_one_step(batch, model, prototype, device)

    x = x.permute(0, 3, 1, 2).flatten(0, 1).flatten(-2, -1).T
    y = y.permute(0, 3, 1, 2).flatten(0, 1).flatten(-2, -1).T

    min_val = min(x.min().item(), y.min().item())
    max_val = max(x.max().item(), y.max().item())

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    lx = ax[0].imshow(x.cpu(), vmin=min_val, vmax=max_val)
    ax[0].set_title("Predicted")
    ly = ax[1].imshow(y.cpu(), vmin=min_val, vmax=max_val)
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
    device,
    rank,
    pbar,
    world_size,
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

    

def main_worker(rank, world_size, args):
    print(f"[main_worker] rank={rank}, world_size={world_size}")
    # DDP setup
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)

    print(f"[main_worker] DDP initialized on rank={rank}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        wandb.init(project="speech_prior", config=vars(args))

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

    collator = SeriesCollator(args.hop_length)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator.collate_batch,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        collate_fn=collator.collate_batch,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model(config)
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

        if epoch % 10 == 0:
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

            model.train()
            prototype.train()

    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train a Wav2Vec2 model with custom dataset")
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument("--hdf5_dir", type=str, default="./datasets")
    parser.add_argument("--features_parquet", type=str, default="features.parquet")
    parser.add_argument("--utterance_parquet", type=str, default="utterance.parquet")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_bins", type=int, default=4000)
    parser.add_argument("--num_folds", type=int, default=64)
    parser.add_argument("--hop_length", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count() if torch.cuda.is_available() else 1)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")

    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

