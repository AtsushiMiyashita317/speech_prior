import argparse

import torch
from tqdm import tqdm
from transformers import Wav2Vec2Config, Wav2Vec2Model, get_linear_schedule_with_warmup

from prior.dataset.hdf5_dataset import HDF5Dataset
from prior.utils.data import SeriesCollator, RandomFoldedLengthBatchSampler
from prior.nn.functional import series_covariance, series_covariance_mask
from prior.nn.model.cnn1d import CNN1dKernel


def main():
    parser = argparse.ArgumentParser(description="Train a Wav2Vec2 model with custom dataset")
    parser.add_argument("base_dir", type=str)
    parser.add_argument("hdf5_dir", type=str)
    parser.add_argument("--features_parquet", type=str, default="features.parquet")
    parser.add_argument("--utterance_parquet", type=str, default="utterance.parquet")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--budget_bytes", type=int, default=0)
    parser.add_argument("--batch_bins", type=int, default=4000)
    parser.add_argument("--num_folds", type=int, default=64)
    parser.add_argument("--hop_length", type=int, default=320)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = HDF5Dataset(
        args.base_dir,
        args.hdf5_dir,
        args.features_parquet,
        args.utterance_parquet,
        cache_dir=args.cache_dir,
        budget_bytes=args.budget_bytes,
    )

    sampler = RandomFoldedLengthBatchSampler(
        train_dataset.utt_lens,
        args.batch_bins,
        args.num_folds
    )
    collator = SeriesCollator(args.hop_length)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collator.collate_batch,
        num_workers=40,
        pin_memory=False
    )

    # --- 1. モデルアーキテクチャの定義 ---
    # "facebook/wav2vec2-base" の設定（アーキテクチャ）を読み込む
    # これにより、重みは読み込まずに構造だけを定義できる
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")

    # ランダムな重みでモデルを初期化
    model = Wav2Vec2Model(config)

    # モデルを学習モードにする
    model.train()

    prototype = CNN1dKernel(num_layers=3, kernel_size=5)

    print("モデルの準備ができました:", model.__class__.__name__)

    # --- 3. オプティマイザとスケジューラの設定 ---
    # AdamWオプティマイザを設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    # ウォームアップのステップ数 (例: 総ステップの10%)
    num_warmup_steps = int(0.1 * num_training_steps)

    # ウォームアップ付き線形減衰スケジューラ
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print("\nオプティマイザとスケジューラを設定しました。")
    print("Optimizer:", optimizer.__class__.__name__)
    print("Scheduler:", scheduler.__class__.__name__)

    model.to(device)
    prototype.to(device)
    pbar1 = tqdm(total=num_epochs)
    pbar2 = tqdm(total=len(train_dataloader))
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            wave, teacher, wave_mask, teacher_mask = batch
            teacher = teacher.to(device)
            b = wave.size(0)
            n = 80 // b
            cov_teacher = series_covariance(teacher, n)
            wave = wave.to(device)
            wave_mask = wave_mask.to(device)
            feature = model.forward(wave, attention_mask=wave_mask).last_hidden_state

            cov_mask = series_covariance_mask(teacher_mask.to(device), n)
            cov_feature = series_covariance(feature, n) * cov_mask

            kernel = torch.stack([cov_feature, cov_feature], dim=-1)
            ntk = prototype.forward(kernel).select(-1, 1)

            x = ntk.masked_select(cov_mask)
            y = cov_teacher.masked_select(cov_mask)

            loss = torch.nn.functional.mse_loss(x, y)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar2.update(1)
            pbar2.set_postfix(loss=loss.item())
        pbar2.reset()
        pbar1.update(1)
