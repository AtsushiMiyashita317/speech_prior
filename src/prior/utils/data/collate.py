import torch

def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad/stack Dataset samples into a batch dict.

    - wave -> [B, S_max]
    - features -> [B, T_max, D]
    """

    B = len(batch)
    # audio
    s_lens = [b[0].shape[0] for b in batch]
    f_lens = [b[1].shape[0] for b in batch]
    Smax = max(s_lens)
    Fmax = max(f_lens)
    D = batch[0][1].shape[1]
    batch_wave = torch.zeros((B, Smax))
    batch_feature = torch.zeros((B, Fmax, D))
    for i, b in enumerate(batch):
        wave, features = b
        batch_wave[i, :wave.shape[0]].copy_(wave)
        batch_feature[i, :features.shape[0], :].copy_(features)

    return batch_wave, batch_feature
