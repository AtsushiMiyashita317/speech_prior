#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合ダンプ: SSL(wav2vec2/HuBERT/WavLM) / Whisper(Encoder) / x-vector
- 再帰探索、モデル別サブディレクトリ、指定ディレクトリ名からの相対ミラー、進捗バー
- 出力: .npy + .meta.json (fp16)
- Whisper は encoder 固定、--layers の 0始まり index を encoder block に対応
"""

import argparse, json, re, hashlib, datetime, platform, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from transformers import (
    AutoModel, AutoProcessor, AutoFeatureExtractor, Wav2Vec2Processor,
    WhisperModel, WhisperProcessor, __version__ as hf_version,
)

# ---------- utils ----------

def sanitize_for_dir(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name)

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

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
    processor = WhisperProcessor.from_pretrained(model_id)
    return model, processor

def build_xvector_model(model_id: str, device: torch.device):
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError as e:
        raise RuntimeError("x-vector を使うには `pip install speechbrain` が必要です") from e
    clf = EncoderClassifier.from_hparams(source=model_id, run_opts={"device": str(device)})
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
    root: Path,
    input_dir: Path,
    model_obj,
    processor,
    model_id: str,
    layer_ids,
    script_version: str,
    device: torch.device,
    overwrite: bool,
    model_subdir: bool,
    show_inner_bar: bool,
    # meta defaults
    fps: int,
    frame_hop_ms: float,
    hop_samples: int,
    window_samples: int,
    subsampling: int,
    frontend: str,
):
    # 出力ディレクトリ計算
    audio_rel_root = str(wav_path.resolve().relative_to(root))
    audio_rel = wav_path.resolve().relative_to(input_dir)

    subset = input_dir.name

    model_name = sanitize_for_dir(model_id)
    # 新仕様: ダンプファイルのパスを /path/to/audio/<model_dir>.<lname>.npy に変更
    out_subdir = outdir / subset / audio_rel.parent
    out_subdir.mkdir(parents=True, exist_ok=True)

    def out_npy(lname: str) -> Path:
        return out_subdir / f"{model_name}.{lname}.npy"

    # 既存チェック
    if not overwrite and arch != "raw":
        if arch in ("ssl", "whisper"):
            targets = [f"layer{lid}" for lid in layer_ids]
        else:  # xvector
            targets = [f"blocks.{lid*3}" for lid in layer_ids]
        if targets and all(out_npy(t).exists() for t in targets):
            return {"status": "skip", "path": str(wav_path)}

    # 音声
    wav, sr = load_audio(wav_path)
    num_samples = len(wav); duration_sec = num_samples / sr
    audio_sha = sha256sum(wav_path)
    fw = {"torch": torch.__version__, "transformers": hf_version}
    device_name = str(device)

    if arch == "raw":
        # wav保存: /path/to/audio/<model_dir>.raw.wav
        raw_path = out_subdir / "raw.wav"
        sf.write(raw_path, wav, sr, subtype="FLOAT")
        meta = {
            "arch": arch,
            "model": None,
            "model_id": None,
            "layer": "raw",
            "dtype_saved": "float32",
            "shape": list(wav.shape),
            "fps": None,
            "frame_hop_ms": None,
            "sample_rate": sr,
            "num_samples": num_samples,
            "duration_sec": duration_sec,
            "hop_samples": None,
            "window_samples": None,
            "subsampling": None,
            "frontend": None,
            "audio_path_rel": audio_rel_root,
            "audio_sha256": audio_sha,
            "dump_path_rel": str(raw_path.resolve().relative_to(root)),
            "script_version": script_version,
            "framework_versions": fw,
            "device": device_name,
            "created_at": datetime.datetime.now().isoformat(),
            "system": platform.platform(),
        }
        with open(str(raw_path).replace(".wav", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return {"status": "ok", "path": str(wav_path)}
    elif arch == "ssl":
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        acts, handles = attach_ssl_hooks(model_obj, layer_ids)
        with torch.no_grad(): _ = model_obj(**inputs)
        for h in handles: h.remove()

        it = layer_ids if not show_inner_bar else tqdm(layer_ids, desc=f"{wav_path.name}", leave=False)
        for lid in it:
            lname = f"layer{lid}"
            feat = acts[lname]
            arr = feat[0] if feat.dim() == 3 and feat.shape[0] == 1 else feat
            if arr.dim() == 3 and arr.shape[1] == 1: arr = arr[:,0,:]
            arr = arr.to(torch.float16).numpy()
            npy_path = out_npy(lname); np.save(npy_path, arr)

            meta = {
                "arch": arch, "model": getattr(model_obj, "name_or_path", None) or model_id, "model_id": model_id,
                "layer": lname, "dtype_saved": "fp16", "dtype_compute": "fp32", "shape": list(arr.shape),
                "fps": fps, "frame_hop_ms": frame_hop_ms, "sample_rate": sr, "num_samples": num_samples,
                "duration_sec": duration_sec, "hop_samples": hop_samples, "window_samples": window_samples,
                "subsampling": subsampling, "frontend": frontend,
                "audio_path_rel": audio_rel_root, "audio_sha256": audio_sha,
                "dump_path_rel": str(npy_path.resolve().relative_to(root)),
                "script_version": script_version, "framework_versions": fw, "device": device_name,
                "created_at": datetime.datetime.now().isoformat(), "system": platform.platform(),
            }
            with open(str(npy_path).replace(".npy", ".meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

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

        it = layer_ids if not show_inner_bar else tqdm(layer_ids, desc=f"{wav_path.name}", leave=False)
        for lid in it:
            idx = 1 + lid  # 0-based block -> hidden_states index
            if idx >= len(hs_tuple):
                raise IndexError(f"requested Whisper layer {lid} but model has only {len(hs_tuple)-1} encoder blocks")
            feat = hs_tuple[idx]  # [B,T,D]
            arr = feat[0] if feat.dim() == 3 and feat.shape[0] == 1 else feat
            arr = arr.to(torch.float16).cpu().numpy()  # [T,D]
            lname = f"layer{lid}"
            npy_path = out_npy(lname); np.save(npy_path, arr)

            meta = {
                "arch": arch, "model": getattr(model_obj, "name_or_path", None) or model_id, "model_id": model_id,
                "layer": lname, "dtype_saved": "fp16", "dtype_compute": "fp32", "shape": list(arr.shape),
                # Whisper 時間解像度（mel 10ms）
                "fps": fps, "frame_hop_ms": frame_hop_ms, "sample_rate": sr,
                "num_samples": num_samples, "duration_sec": duration_sec,
                "hop_samples": hop_samples, "window_samples": window_samples,
                "subsampling": subsampling, "frontend": "mel80_10ms",
                "audio_path_rel": audio_rel_root, "audio_sha256": audio_sha,
                "dump_path_rel": str(npy_path.resolve().relative_to(root)),
                "script_version": script_version, "framework_versions": fw, "device": device_name,
                "created_at": datetime.datetime.now().isoformat(), "system": platform.platform(),
            }
            with open(str(npy_path).replace(".npy", ".meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    elif arch == "xvector":
        # fbank frontend
        try:
            import torchaudio
        except ImportError as e:
            raise RuntimeError("x-vector には torchaudio が必要です: pip install torchaudio") from e
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(device)  # [1, n]
        # 長さは正規化用に相対長でOK（1.0）※音声ごとに可変でもよい
        wav_lens = torch.ones(wav_t.shape[0], device=device)

        # 1) SBフロントエンドで特徴抽出（モデルに合った次元=ここでは24）
        feats = model_obj.mods.compute_features(wav_t)          # [B, T, F]  (F=24想定)
        feats = model_obj.mods.mean_var_norm(feats, wav_lens)   # [B, T, F]  正規化

        acts, handles = attach_xvector_hooks(model_obj, layer_ids)
        with torch.no_grad():
            em = model_obj.mods.embedding_model
            _ = em(feats)  # ← これで最初のConv(in=24)に合う
        for h in handles: h.remove()

        lnames = [f"blocks.{lid*3}" for lid in layer_ids]
        it = lnames if not show_inner_bar else tqdm(lnames, desc=f"{wav_path.name}", leave=False)
        for lname in it:
            feat = acts[lname]  # [B,T,C]
            arr = feat[0] if feat.dim() == 3 and feat.shape[0] == 1 else feat
            arr = arr.to(torch.float16).numpy()
            npy_path = out_npy(lname); np.save(npy_path, arr)

            # hparams は SimpleNamespace のことがある（SB 1.x）。安全に source を拾う
            src_model_id = model_id
            hp = getattr(model_obj, "hparams", None)
            if hp is not None:
                # dict 互換 or 属性の両対応
                if isinstance(hp, dict):
                    src_model_id = hp.get("source", model_id) or model_id
                else:
                    src_model_id = getattr(hp, "source", None) or model_id

            meta = {
                "arch": arch, "model": src_model_id,
                "model_id": model_id, "layer": lname, "dtype_saved": "fp16", "dtype_compute": "fp32",
                "shape": list(arr.shape),
                "fps": fps, "frame_hop_ms": frame_hop_ms, "sample_rate": sr,
                "num_samples": num_samples, "duration_sec": duration_sec,
                "hop_samples": hop_samples, "window_samples": 400, "subsampling": subsampling,
                "frontend": "fbank10ms_80",
                "audio_path_rel": audio_rel_root, "audio_sha256": audio_sha,
                "dump_path_rel": str(npy_path.resolve().relative_to(root)),
                "script_version": script_version, "framework_versions": fw, "device": device_name,
                "created_at": datetime.datetime.now().isoformat(), "system": platform.platform(),
            }
            with open(str(npy_path).replace(".npy", ".meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    else:
        raise ValueError(f"unknown arch: {arch}")

    return {"status": "ok", "path": str(wav_path)}

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--outdir", default="dumps")
    ap.add_argument("--root", default=".")
    ap.add_argument("--arch", choices=["ssl","whisper","xvector","raw"], default="ssl")
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--layers", default="6,9,11", help="ssl/whisper: 0始まりの層番号（カンマ区切り）")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_model_subdir", action="store_true")
    ap.add_argument("--exts", default=".flac", help="探索拡張子（例: .flac,.wav）")
    # メタ上書き
    ap.add_argument("--fps", type=int)
    ap.add_argument("--hop_samples", type=int)
    ap.add_argument("--subsampling", type=int)
    ap.add_argument("--frame_hop_ms", type=float)
    ap.add_argument("--window_samples", type=int, default=400)
    ap.add_argument("--script_version", default="dump_features@1.1")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root).resolve()
    exts = tuple(x.strip() for x in args.exts.split(",") if x.strip())

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDAが利用できないためCPUで実行します。")

    # デフォの fps/hop/subsampling/frontend
    if args.arch == "ssl":
        defaults = dict(fps=50, hop_samples=320, subsampling=320, frame_hop_ms=20.0, frontend="raw16k")
        model_obj, processor = build_ssl_model(args.model, device)
        layer_ids = [int(x) for x in args.layers.split(",")]
    elif args.arch == "whisper":
        defaults = dict(fps=100, hop_samples=160, subsampling=1, frame_hop_ms=10.0, frontend="mel80_10ms")
        model_obj, processor = build_whisper_model(args.model, device)
        layer_ids = [int(x) for x in args.layers.split(",")]
    elif args.arch == "xvector":
        defaults = dict(fps=100, hop_samples=160, subsampling=1, frame_hop_ms=10.0, frontend="fbank10ms_80")
        model_obj = build_xvector_model(args.model, device)
        processor = None
        layer_ids = [int(x) for x in args.layers.split(",")]
    elif args.arch == "raw":
        defaults = dict(fps=16000, hop_samples=160, subsampling=1, frame_hop_ms=10.0, frontend="raw16k")
        model_obj = None
        processor = None
        layer_ids = None
    else:
        raise ValueError(args.arch)

    fps = args.fps or defaults["fps"]
    hop_samples = args.hop_samples or defaults["hop_samples"]
    subsampling = args.subsampling or defaults["subsampling"]
    frame_hop_ms = args.frame_hop_ms or defaults["frame_hop_ms"]
    window_samples = args.window_samples
    frontend = defaults["frontend"]

    files = list(iter_audio_files(input_dir, exts))
    if not files:
        print(f"[note] 音声が見つかりませんでした: {input_dir} (exts={exts})")
        return

    model_subdir = not args.no_model_subdir
    err_log = outdir / "errors.log"
    if err_log.exists(): err_log.unlink()

    def task(p: Path):
        try:
            return dump_one_file(
                p,
                arch=args.arch,
                outdir=outdir,
                root=root,
                input_dir=input_dir,
                model_obj=model_obj,
                processor=processor,
                model_id=args.model,
                layer_ids=layer_ids,
                script_version=args.script_version,
                device=device,
                overwrite=args.overwrite,
                model_subdir=model_subdir,
                show_inner_bar=(args.num_workers == 1),
                fps=fps, frame_hop_ms=frame_hop_ms, hop_samples=hop_samples,
                window_samples=window_samples, subsampling=subsampling, frontend=frontend,
            )
        except Exception as e:
            with open(err_log, "a", encoding="utf-8") as f:
                f.write(f"{p}\n{traceback.format_exc()}\n")
            return {"status": "error", "path": str(p), "error": str(e)}

    results = {"ok": 0, "skip": 0, "error": 0}
    if args.num_workers == 1:
        for p in tqdm(files, desc="files"):
            r = task(p)
            results[r["status"]] = results.get(r["status"], 0) + 1
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            futs = [ex.submit(task, p) for p in files]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="files"):
                r = fut.result()
                results[r["status"]] = results.get(r["status"], 0) + 1

    print(f"[done] ok={results['ok']} skip={results['skip']} error={results['error']}")
    if results["error"] > 0:
        print(f"[note] エラー詳細: {err_log}")

if __name__ == "__main__":
    main()
