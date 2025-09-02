import torch
from typing import Optional, Union
from transformers import Wav2Vec2Model

class BothsidePaddedWav2Vec2Model(Wav2Vec2Model):
    """
    Wav2Vec2Modelを継承し、`_get_feature_vector_attention_mask`メソッドを
    前方パディングに対応させたカスタムモデルクラス。
    """
    def _get_feat_extract_output_pad(
        self, pad_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length, stride, rounding_mode="floor")

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            pad_lengths = _conv_out_length(pad_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                pad_lengths = _conv_out_length(pad_lengths, 1, self.config.adapter_stride)

        return pad_lengths
    
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # 1. 入力のattention_maskから、パディングを除いた実際の入力長を計算
        mask_cumsum = attention_mask.cumsum(-1).to(torch.long)
        input_lengths = mask_cumsum.select(-1, -1)
        input_left_pad = mask_cumsum.eq(0).sum(-1)
        
        # 2. モデルの内部関数を使い、CNN特徴抽出後の実際の出力長を計算
        output_lengths = self._get_feat_extract_output_lengths(input_lengths, add_adapter=add_adapter)
        output_left_pad = self._get_feat_extract_output_pad(input_left_pad, add_adapter=add_adapter)

        # 3. 前方パディングを考慮したマスクを生成
        batch_size = attention_mask.shape[0]
        
        # マスクをすべて0で初期化
        output_mask = torch.zeros(
            (batch_size, feature_vector_length), 
            dtype=torch.long, 
            device=attention_mask.device
        )
        
        # 各サンプルの後方を、計算した出力長ぶんだけ1で埋める
        for i, (length, pad) in enumerate(zip(output_lengths, output_left_pad)):
            output_mask[i, pad:pad+length] = 1
            
        return output_mask.bool()
    