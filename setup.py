# setup.py
from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
readme = (this_dir / "README.md").read_text(encoding="utf-8") if (this_dir / "README.md").exists() else ""

setup(
    name="speech_prior",
    version="0.1.0",
    description="Unified dumper for wav2vec2 / HuBERT / WavLM / Conformer / x-vector frame-level features",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="(your name)",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    # まずは scripts/ から素直に入れる。あとで console_scripts に切替OK
    install_requires=[
        # core
        "numpy>=1.24",
        "soundfile>=0.12.1",     # flac/wav ロード
        "tqdm>=4.66",
        # model stacks
        "torch>=2.2",            # CUDA 版は環境に合わせて別途
        "torchaudio>=2.2",
        "transformers>=4.42.0",  # wav2vec2 / HuBERT / WavLM
        # optional frontends / models（必要に応じて後で追加）
        "speechbrain>=0.5.14", 
        "huggingface_hub>=0.20", 
        # storage
        "zarr>=2.16.1",        # Zarr 保存に切替えるなら
        "h5py>=3.11.0",        # HDF5 を使うなら
        "pyarrow>=16.1.0",     # Parquet/Arrow を使うなら
        "polars>=1.5.0",       # 索引/メタ管理を列指向で
        "duckdb>=1.0.0",
    ],
    extras_require={
        "ssl": ["transformers>=4.42.0"],
        "xvector": ["speechbrain>=1.0.0"],
        "zarr": ["zarr>=2.16.1", "numcodecs>=0.12.1"],
        "hdf5": ["h5py>=3.11.0"],
        "arrow": ["pyarrow>=16.1.0"],
        "analytics": ["polars>=1.5.0", "duckdb>=1.0.0"],
        "dev": ["black>=24.4.2", "ruff>=0.5.0", "pytest>=8.2.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Source": "https://example.com/your-repo",  # 後で差し替え
        "Issues": "https://example.com/your-repo/issues",
    },
)
