import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

import torch

from src.data import DATASET_REPO, DATASET_REVISION

PACKAGES = ["torch", "transformers", "tokenizers", "accelerate", "datasets", "safetensors", "sentencepiece",
            "protobuf", "normalizer", "numpy", "pandas", "scikit-learn"]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def cpu_name() -> str:
    for line in open("/proc/cpuinfo", encoding="utf-8"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def environment() -> dict:
    info = {
        "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"])),
        "python": platform.python_version(),
        "packages": package_versions(),
        "torch_cuda": torch.version.cuda,
        "cpu": cpu_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["driver"] = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0]
    return info
