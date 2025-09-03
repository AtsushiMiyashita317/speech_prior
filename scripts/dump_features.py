#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合ダンプ: SSL(wav2vec2/HuBERT/WavLM) / Whisper(Encoder) / x-vector
- 再帰探索、モデル別サブディレクトリ、指定ディレクトリ名からの相対ミラー、進捗バー
- 出力: .npy + .meta.json (fp16)
- Whisper は encoder 固定、--layers の 0始まり index を encoder block に対応
"""

import argparse, traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from transformers import (
    AutoModel, AutoProcessor, AutoFeatureExtractor, Wav2Vec2Processor,
    WhisperModel, WhisperProcessor
)
from speechbrain.inference.classifiers import EncoderClassifier

MAX_LEN = 1500

# ---------- utils ----------

def load_audio(path: Path, target_sr=16000):
    x, sr = sf.read(str(path))
    if sr != target_sr:
        raise ValueError(f"{path}: expected {target_sr} Hz, got {sr}")
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    return x.astype(np.float32), sr

def iter_audio_files(input_dir: Path, exts):
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p
            
def ssl_effective_lengths(length: float):
    T_hid_eff = (length - 400) // 320 + 1
    return T_hid_eff

# ---------- builders ----------

def build_ssl_model(model_id: str, device: torch.device):
    # Processor
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_id)
        except Exception:
            fe = AutoFeatureExtractor.from_pretrained(model_id)
            class _Wrapper:
                def __init__(self, fe): self.fe = fe
                def __call__(self, x, sampling_rate, return_tensors="pt"):
                    return self.fe(x, sampling_rate=sampling_rate, return_tensors=return_tensors)
            processor = _Wrapper(fe)
    # Model
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return model, processor

def build_whisper_model(model_id: str, device: torch.device):
    model = WhisperModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def build_xvector_model(model_id: str, device: torch.device):
    clf = EncoderClassifier.from_hparams(source=model_id, run_opts={"device": str(device)}).eval()
    return clf

# ---------- hooks ----------

def attach_ssl_hooks(model, layer_ids):
    acts = {}
    def make_hook(name):
        def _hook(module, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            acts[name] = x.detach().cpu()  # [B,T,D]
        return _hook
    handles = [model.encoder.layers[lid].register_forward_hook(make_hook(f"layer{lid}"))
               for lid in layer_ids]
    return acts, handles

def attach_xvector_hooks(xvec_model, layer_ids):
    acts, handles = {}, []
    em = getattr(xvec_model.mods, "embedding_model", None)
    if em is None:
        raise RuntimeError("unexpected x-vector model structure: embedding_model not found")

    def make_hook(name):
        def _hook(module, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            acts[name] = x.detach().cpu()
        return _hook

    for lid in layer_ids:
        mod = getattr(em, "blocks", None)[lid*3]
        if mod is None:
            raise RuntimeError(f"x-vector layer not found: blocks.{lid*3}")
        handles.append(mod.register_forward_hook(make_hook(f"blocks.{lid*3}")))
    return acts, handles

# ---------- core dump ----------

def dump_one_file(
    wav_path: Path,
    *,
    arch: str,
    outdir: Path,
    input_dir: Path,
    model_obj,
    processor,
    layer_ids,
    device: torch.device,
    overwrite: bool,
):
    # 音声
    wav, sr = load_audio(wav_path)
    num_samples = len(wav)
    ssl_len = ssl_effective_lengths(num_samples)
    if ssl_len > MAX_LEN:
        return {"status": "skip", "path": str(wav_path)}
    
    # 出力ディレクトリ計算
    audio_rel = wav_path.resolve().relative_to(input_dir)

    subset = input_dir.name
    filename = audio_rel.stem

    out_subdir = outdir / subset / audio_rel.parent / filename
    out_subdir.mkdir(parents=True, exist_ok=True)

    def out_npy(lid: int) -> Path:
        return out_subdir / f"{arch}.{lid}.npy"

    if not overwrite:
        if layer_ids and all(out_npy(lid).exists() for lid in layer_ids):
            return {"status": "skip", "path": str(wav_path)}

    if arch in ["hubert", "wavlm", "wav2vec"]:
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        acts, handles = attach_ssl_hooks(model_obj, layer_ids)
        with torch.no_grad(): _ = model_obj(**inputs)
        for h in handles: h.remove()

        it = layer_ids
        for lid in it:
            lname = f"layer{lid}"
            feat = acts[lname]
            arr = feat[0] if feat.dim() == 3 and feat.shape[0] == 1 else feat
            if arr.dim() == 3 and arr.shape[1] == 1: arr = arr[:,0,:]
            arr = arr.to(torch.float16).numpy()
            assert arr.shape[0] == ssl_len, f"shape mismatch: {arr.shape[0]} != {ssl_len}"
            npy_path = out_npy(lid); np.save(npy_path, arr)

    elif arch == "whisper":
        # Whisper: encoder 固定、--layers は encoder block の 0始まり
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            encoder = model_obj.get_encoder()
            enc_out = encoder(
                input_features=inputs["input_features"].to(device),
                output_hidden_states=True,
                return_dict=True,
            )
        hs_tuple = enc_out.hidden_states  # tuple[L+1], 0は入力埋め込み

        it = layer_ids
        for lid in it:
            idx = 1 + lid  # 0-based block -> hidden_states index
            if idx >= len(hs_tuple):
                raise IndexError(f"requested Whisper layer {lid} but model has only {len(hs_tuple)-1} encoder blocks")
            feat = hs_tuple[idx]  # [B,T,D]
            arr = feat[0] if feat.dim() == 3 and feat.shape[0] == 1 else feat
            arr = arr[:ssl_len]  # [T,D]
            arr = arr.to(torch.float16).cpu().numpy()  # [T,D]
            npy_path = out_npy(lid); np.save(npy_path, arr)

    elif arch == "xvector":
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(device)  # [1, n]

        acts, handles = attach_xvector_hooks(model_obj, layer_ids)
        with torch.no_grad():
            _ = model_obj.encode_batch(wav_t)
        for h in handles: h.remove()

        it = layer_ids
        for lid in it:
            lname = f"blocks.{lid*3}"
            feat = acts[lname]  # [B,T,C]
            arr = feat[0] if feat.dim() == 3 and feat.shape[0] == 1 else feat
            arr = arr[:ssl_len*2:2]
            arr = arr.to(torch.float16).numpy()
            npy_path = out_npy(lid); np.save(npy_path, arr)

    else:
        raise ValueError(f"unknown arch: {arch}")

    return {"status": "ok", "path": str(wav_path)}

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--outdir", default="dumps")
    ap.add_argument("--arch", choices=["hubert","wavlm","wav2vec","whisper","xvector"], default="wav2vec")
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--layers", default="6,9,11", help="ssl/whisper: 0始まりの層番号(カンマ区切り)")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exts", default=".flac", help="探索拡張子（例: .flac,.wav)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    exts = tuple(x.strip() for x in args.exts.split(",") if x.strip())

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDAが利用できないためCPUで実行します。")

    # デフォの fps/hop/subsampling/frontend
    if args.arch in ["hubert","wavlm","wav2vec","ssl"]:
        model_obj, processor = build_ssl_model(args.model, device)
        layer_ids = [int(x) for x in args.layers.split(",")]
    elif args.arch == "whisper":
        model_obj, processor = build_whisper_model(args.model, device)
        layer_ids = [int(x) for x in args.layers.split(",")]
    elif args.arch == "xvector":
        model_obj = build_xvector_model(args.model, device)
        processor = None
        layer_ids = [int(x) for x in args.layers.split(",")]
    else:
        raise ValueError(args.arch)

    files = list(iter_audio_files(input_dir, exts))
    if not files:
        print(f"[note] 音声が見つかりませんでした: {input_dir} (exts={exts})")
        return

    err_log = outdir / "errors.log"
    if err_log.exists(): err_log.unlink()

    def task(p: Path):
        try:
            return dump_one_file(
                p,
                arch=args.arch,
                outdir=outdir,
                input_dir=input_dir,
                model_obj=model_obj,
                processor=processor,
                layer_ids=layer_ids,
                device=device,
                overwrite=args.overwrite,
            )
        except Exception as e:
            with open(err_log, "a", encoding="utf-8") as f:
                f.write(f"{p}\n{traceback.format_exc()}\n")
            return {"status": "error", "path": str(p), "error": str(e)}

    results = {"ok": 0, "skip": 0, "error": 0}
    for p in tqdm(files, desc="files"):
        r = task(p)
        results[r["status"]] = results.get(r["status"], 0) + 1

    print(f"[done] ok={results['ok']} skip={results['skip']} error={results['error']}")
    if results["error"] > 0:
        print(f"[note] エラー詳細: {err_log}")

if __name__ == "__main__":
    main()
