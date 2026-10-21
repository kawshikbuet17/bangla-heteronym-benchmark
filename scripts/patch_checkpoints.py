"""Adds the dataset reading fix (src.data.READING_SWAPS) to checkpoints trained on the unfixed dataset revision.

The models keep predicting what they learned (for ব্যাংককে, the swapped reading); `reading_fixes` in
heteronym_config.json maps their output to the correct reading at prediction time (src.predict and
bangla_heteronym_package). Only for checkpoints trained before the fix (all checkpoints of 2026-10-18..21).

    python -m scripts.patch_checkpoints                       # every heteronym_config.json under checkpoints/
    python -m scripts.patch_checkpoints checkpoints/banglabert
"""
import json
import sys
from pathlib import Path

from src.data import READING_SWAPS

FIXES = {word: {a: b, b: a} for word, (a, b) in READING_SWAPS.items()}


def patch(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["reading_fixes"] = FIXES
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"patched {config_path}")


def main() -> None:
    roots = [Path(p) for p in sys.argv[1:]] or [Path("checkpoints")]
    for root in roots:
        for config_path in sorted(root.rglob("heteronym_config.json")):
            patch(config_path)


if __name__ == "__main__":
    main()
