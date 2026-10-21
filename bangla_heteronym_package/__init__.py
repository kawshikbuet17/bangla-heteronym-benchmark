"""Bangla heteronym reader: replaces heteronyms such as কর with their reading (কর্‌ / করো / কোরো) in context.

    import bangla_heteronym_maker as bhm
    bhm.initialize("/abs/path/to/checkpoints/banglabert")    # once; later calls return the same reader
    reader = bhm.get_reader()                                 # anywhere else in the program
    reader.phoneticize("তুমি একটু অপেক্ষা কর, আমি আসছি।")      # 'তুমি একটু অপেক্ষা করো, আমি আসছি।'
"""
import os
import threading
from pathlib import Path

from .interface import HeteronymReader, Reading
from .reader import HeteronymModelReader

__all__ = ["HeteronymReader", "HeteronymModelReader", "Reading", "initialize", "get_reader", "is_initialized", "reset",
           "MODEL_PATH_ENV"]

MODEL_PATH_ENV = "BANGLA_HETERONYM_MODEL"  # used by initialize() when no path is given

_lock = threading.Lock()
_reader: HeteronymModelReader | None = None
_path: Path | None = None


def initialize(checkpoint: str | Path | None = None, device: str | None = None, precision: str | None = None,
               batch_size: int = 64) -> HeteronymReader:
    """Loads the model once per process (thread-safe singleton) and returns the reader.

    checkpoint: path to a checkpoint folder (absolute, or relative to the current directory); if None, the
    BANGLA_HETERONYM_MODEL environment variable. device: "cuda", "cuda:1" or "cpu" (default: cuda if available).
    precision: "fp16" or "fp32" (default: training precision on GPU, fp32 on CPU). batch_size: occurrences per model
    call in the batch methods. Calling it again returns the same reader; asking for a different checkpoint raises
    RuntimeError (call reset() first to switch models).
    """
    global _reader, _path
    checkpoint = checkpoint or os.environ.get(MODEL_PATH_ENV)
    if not checkpoint:
        raise ValueError(f"give the checkpoint path or set {MODEL_PATH_ENV}")
    path = Path(checkpoint).expanduser().resolve()
    with _lock:
        if _reader is None:
            _reader = HeteronymModelReader.load(path, device, precision, batch_size)
            _path = path
        elif path != _path:
            raise RuntimeError(f"already initialized with {_path}; call reset() before loading {path}")
        return _reader


def get_reader() -> HeteronymReader:
    """The reader created by initialize(); initializes from BANGLA_HETERONYM_MODEL if that is set."""
    if _reader is None:
        if os.environ.get(MODEL_PATH_ENV):
            return initialize()
        raise RuntimeError("call bangla_heteronym_maker.initialize(path) first")
    return _reader


def is_initialized() -> bool:
    return _reader is not None


def reset() -> None:
    """Drops the loaded model (frees its memory once no other reference is held)."""
    global _reader, _path
    with _lock:
        _reader, _path = None, None
