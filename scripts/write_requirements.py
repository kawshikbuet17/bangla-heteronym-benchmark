"""Writes requirements/{model}.txt with the exact versions installed in this env (what the model was run with)."""
import json
import sys
from importlib.metadata import distribution, version
from pathlib import Path

from huggingface_hub import list_repo_files

from src.train import load_config

BASE = ["transformers", "tokenizers", "accelerate", "safetensors", "huggingface_hub", "datasets", "numpy", "pandas",
        "scikit-learn", "PyYAML", "tensorboard"]


def normalizer_line() -> str:
    url = json.loads(distribution("normalizer").read_text("direct_url.json"))
    return f"normalizer @ git+{url['url']}@{url['vcs_info']['commit_id']}"


model = sys.argv[1]
cfg = load_config(f"configs/{model}.yaml")
torch_version = version("torch")
lines = [f"# {model} ({cfg['model_id']}): packages used to train and run inference",
         f"--extra-index-url https://download.pytorch.org/whl/{torch_version.split('+')[-1]}",
         f"torch=={torch_version.split('+')[0]}"]
lines += [f"{name}=={version(name)}" for name in BASE]
if any(f.endswith(".model") for f in list_repo_files(cfg["model_id"])):
    lines += [f"sentencepiece=={version('sentencepiece')}", f"protobuf=={version('protobuf')}"]
if cfg["normalize"]:
    lines.append(normalizer_line())
Path("requirements", f"{model}.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
