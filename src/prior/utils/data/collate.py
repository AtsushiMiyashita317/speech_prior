import random
import torch

class SeriesCollator:
    def __init__(self, hop_length=320):
        self.hop_length = hop_length

    def collate_batch(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
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
        mask_wave = torch.zeros((B, Smax), dtype=torch.long)
        mask_feature = torch.zeros((B, Fmax), dtype=torch.long)
        for i, b in enumerate(batch):
            wave, features = b
            f = features.shape[0]
            offset_f = random.randint(0, max(0, Fmax - f - 1))
            offset_s = offset_f * self.hop_length
            batch_wave[i, offset_s:offset_s + wave.shape[0]].copy_(wave)
            batch_feature[i, offset_f:offset_f + features.shape[0], :].copy_(features)
            mask_wave[i, offset_s:offset_s + wave.shape[0]] = 1
            mask_feature[i, offset_f:offset_f + features.shape[0]] = 1

        return batch_wave, batch_feature, mask_wave, mask_feature
