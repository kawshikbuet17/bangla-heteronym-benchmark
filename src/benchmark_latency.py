import argparse
import gc
import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data import LATENCY_PARAGRAPH, LATENCY_SENTENCES, label_rows, load_words, read_csv
from src.env import cpu_name, package_versions
from src.models import Item
from src.predict import Predictor

WARMUP = 20
GPU_BATCHES = (8, 32)
CPU_BATCHES = (8,)
THROUGHPUT_REPEATS = 3
PARAGRAPH_REPEATS = 10


def timed(fn: Callable, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def stats(ms: list[float]) -> dict:
    return {"mean": float(np.mean(ms)), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99))}


def bench(checkpoint: str, device: str, items: list[Item], paragraph: str, batches: tuple[int, ...], threads: int | None) -> dict:
    if threads:
        torch.set_num_threads(threads)
    predictor = Predictor.load(checkpoint, device)
    dev = predictor.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for item in items[:WARMUP]:
        predictor.predict_items([item])
    full = [timed(lambda: predictor.predict_items([item]), dev) for item in items]
    encoded = [predictor.encode([item]) for item in items]
    for batch in encoded[:WARMUP]:
        predictor.forward(batch)
    forward = [timed(lambda: predictor.forward(batch), dev) for batch in encoded]

    throughput = {}
    for size in batches:
        predictor.predict_items(items[:size], batch_size=size)
        totals = [timed(lambda: predictor.predict_items(items, batch_size=size), dev) for _ in range(THROUGHPUT_REPEATS)]
        throughput[str(size)] = len(items) / (np.median(totals) / 1000)

    n_heteronyms = len(predictor.phoneticize_detailed(paragraph)[1])
    for _ in range(3):
        predictor.phoneticize(paragraph)
    paragraph_ms = [timed(lambda: predictor.phoneticize(paragraph), dev) for _ in range(PARAGRAPH_REPEATS)]

    result = {
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else cpu_name(),
        "precision": "fp16" if next(predictor.model.parameters()).dtype == torch.float16 else "fp32",
        "threads": torch.get_num_threads() if dev.type == "cpu" else None,
        "timed_calls": len(items), "warmup_calls": WARMUP,
        "batch1_full_ms": stats(full),
        "batch1_forward_ms": stats(forward),
        "throughput_sentences_per_s": throughput,
        "phoneticize": {"heteronyms": n_heteronyms, "repeats": PARAGRAPH_REPEATS, "total_ms": stats(paragraph_ms),
                        "ms_per_heteronym": float(np.median(paragraph_ms)) / n_heteronyms},
    }
    if predictor.kind == "seq2seq":
        result["avg_generated_tokens"] = float(np.mean([p.generated_tokens for p in predictor.predict_items(items)]))
    if dev.type == "cuda":
        result["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    df = label_rows(read_csv(LATENCY_SENTENCES), load_words())
    items = list(zip(df["sentence"], df["start"], df["end"], df["spelling"]))
    paragraph = LATENCY_PARAGRAPH.read_text(encoding="utf-8")
    hconfig = json.loads((Path(args.checkpoint) / "heteronym_config.json").read_text(encoding="utf-8"))
    result = {
        "checkpoint": args.checkpoint, "model_id": hconfig["model_id"], "model_type": hconfig["model_type"],
        "date": datetime.now().isoformat(timespec="seconds"), "packages": package_versions(),
        "gpu": bench(args.checkpoint, "cuda", items, paragraph, GPU_BATCHES, None),
        "cpu_4_threads": bench(args.checkpoint, "cpu", items, paragraph, CPU_BATCHES, 4),
        "cpu_1_thread": bench(args.checkpoint, "cpu", items, paragraph, (), 1),
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("gpu", "cpu_4_threads", "cpu_1_thread"):
        r = result[key]
        print(f"{key}: p50 {r['batch1_full_ms']['p50']:.2f} ms, forward p50 {r['batch1_forward_ms']['p50']:.2f} ms, "
              f"throughput {r['throughput_sentences_per_s']}, paragraph {r['phoneticize']['total_ms']['p50']:.1f} ms")


if __name__ == "__main__":
    main()
