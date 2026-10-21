"""Bangla heteronym reader. See README.md and demo.py.

    import bangla_heteronym_package as bhp
    reader = bhp.initialize("/abs/path/checkpoints/banglabert")   # or initialize() with the settings in .env
    reader.phoneticize("তুমি একটু অপেক্ষা কর, আমি আসছি।")          # 'তুমি একটু অপেক্ষা করো, আমি আসছি।'
"""
import os
import threading
from pathlib import Path

from .interface import HeteronymReader
from .reader import HeteronymModelReader

__all__ = ["HeteronymReader", "HeteronymModelReader", "initialize", "get_reader", "reset"]

_lock = threading.Lock()
_reader = None
_path = None


def _setting(name: str) -> str | None:
    """An environment variable, else the same key in the .env file next to this file."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip("\"'")
    return None


def initialize(checkpoint_path=None, device: str | None = None) -> HeteronymReader:
    """Loads the model once (thread-safe singleton) and returns the reader; later calls return the same reader.

    checkpoint_path: the checkpoint folder (absolute path recommended); default BANGLA_HETERONYM_MODEL.
    device: "cuda", "cuda:1" or "cpu"; default BANGLA_HETERONYM_DEVICE, else cuda if available.
    """
    global _reader, _path
    checkpoint_path = checkpoint_path or _setting("BANGLA_HETERONYM_MODEL")
    if not checkpoint_path:
        raise ValueError("give the checkpoint path or set BANGLA_HETERONYM_MODEL (environment or .env)")
    path = Path(checkpoint_path).expanduser().resolve()
    with _lock:
        if _reader is None:
            _reader, _path = HeteronymModelReader(path, device or _setting("BANGLA_HETERONYM_DEVICE")), path
        elif path != _path:
            raise RuntimeError(f"already initialized with {_path}; call reset() first to load another model")
        return _reader


def get_reader() -> HeteronymReader:
    """The reader from initialize(); initializes from the settings if that was not called yet."""
    return _reader or initialize()


def reset() -> None:
    """Forgets the loaded model, so initialize() can load another one."""
    global _reader, _path
    with _lock:
        _reader, _path = None, None
