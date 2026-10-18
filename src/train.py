import argparse
import csv
import gc
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainerCallback,
    TrainingArguments, set_seed,
)

from src.data import REAL_TEST, label_rows, load_split, load_words, read_csv
from src.env import environment
from src.evaluate import evaluate
from src.models import (
    build_model, candidate_features, collate, encoder_features, get_normalizer, keeps_zwnj, save_checkpoint,
    seq2seq_inputs,
)
from src.predict import Predictor

RUNS = Path("runs")
CHECKPOINT_RUNS = Path("checkpoints/_runs")
SMOKE_DIR = Path("logs/smoke")
SPLITS = {"main": ("train", "dev", "test"), "unseen": ("unseen_train", "unseen_dev", "unseen_test")}
SETTINGS = {
    "main": {},
    "unseen": {"split": "unseen"},
    "lc2": {"train_file": "train_lc2"},
    "lc5": {"train_file": "train_lc5"},
    "lc10": {"train_file": "train_lc10"},
    "cand_main": {"model_type": "candidate"},
    "cand_unseen": {"model_type": "candidate", "split": "unseen"},
    "abl_marker_special": {"marker": ["<h>", "</h>"], "add_marker_tokens": True},
    "abl_marker_none": {"marker": None},
    "abl_head_mean": {"head": "mean"},
    "abl_head_cls": {"head": "cls"},
    "abl_output_target": {"output": "target"},
}
KEEP_CHECKPOINT = {"main", "cand_main"}
SMOKE_STEPS = 20


class NonFiniteLoss(RuntimeError):
    pass


class Monitor(TrainerCallback):
    """Writes train_log.csv, keeps the best weights in memory (no per-epoch checkpoints on disk), stops early
    on dev accuracy and stops on a non-finite loss."""

    def __init__(self, path: Path, patience: int):
        self.path, self.patience = path, patience
        self.rows: list[dict] = []
        self.epoch_losses: list[float] = []
        self.last_lr = None
        self.best, self.best_epoch, self.bad_epochs = -1.0, 0, 0
        self.best_state: dict[str, torch.Tensor] = {}
        self.nonfinite = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.nonfinite |= not math.isfinite(logs["loss"])
            self.epoch_losses.append(logs["loss"])
            self.last_lr = logs.get("learning_rate")
            self.rows.append({"kind": "step", "epoch": round(state.epoch, 3), "step": state.global_step,
                              "train_loss": logs["loss"], "learning_rate": self.last_lr, "grad_norm": logs.get("grad_norm")})
        if self.nonfinite:
            control.should_training_stop = True

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        accuracy = metrics["eval_accuracy"]
        self.nonfinite |= not math.isfinite(metrics["eval_loss"])
        self.rows.append({"kind": "epoch", "epoch": round(state.epoch, 3), "step": state.global_step,
                          "train_loss": float(np.mean(self.epoch_losses)) if self.epoch_losses else None,
                          "dev_loss": metrics["eval_loss"], "dev_accuracy": accuracy, "learning_rate": self.last_lr})
        self.epoch_losses = []
        if accuracy > self.best:
            self.best, self.best_epoch, self.bad_epochs = accuracy, round(state.epoch), 0
            self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
            control.should_training_stop = self.bad_epochs >= self.patience
        if self.nonfinite:
            control.should_training_stop = True
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["lr_grid"] = [float(lr) for lr in cfg["lr_grid"]]
    return cfg


def best_lr(model_name: str) -> float:
    path = RUNS / model_name / "best_lr.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the lr search and scripts/select_lr.py first")
    return json.loads(path.read_text())["lr"]


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20, 1)


def make_features(hconfig: dict, tokenizer, df: pd.DataFrame) -> Dataset:
    items = list(zip(df["sentence"], df["start"], df["end"], df["spelling"]))
    norm = get_normalizer(hconfig["normalize"], hconfig["strip_zwnj"])
    marker = tuple(hconfig["marker"]) if hconfig["marker"] else None
    if hconfig["model_type"] == "seq2seq":
        inputs = tokenizer(seq2seq_inputs(items, marker, norm), truncation=True, max_length=hconfig["max_length"])
        column = "sentence_phonetic" if hconfig["output"] == "sentence" else "pronunciation"
        labels = tokenizer(text_target=[norm(t) for t in df[column]], truncation=True, max_length=hconfig["max_length"])
        return Dataset.from_dict({**inputs, "labels": labels["input_ids"]})
    build = candidate_features if hconfig["model_type"] == "candidate" else encoder_features
    features = build(tokenizer, items, hconfig["candidates"], marker, norm, hconfig["max_length"])
    for feature, label in zip(features, df["label"]):
        feature["labels"] = int(label)
    return Dataset.from_list(features)


def fit(cfg: dict, train_df: pd.DataFrame, dev_df: pd.DataFrame, run_dir: Path, ckpt_dir: Path, smoke: bool) -> dict:
    set_seed(cfg["seed"])
    words = load_words()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], **cfg.get("tokenizer_kwargs", {}))
    if cfg.get("add_marker_tokens"):
        tokenizer.add_special_tokens({"additional_special_tokens": list(cfg["marker"])})
    hconfig = {k: cfg.get(k) for k in ("model_type", "model_id", "marker", "normalize", "max_length", "head", "dropout", "output", "precision")}
    hconfig["strip_zwnj"] = cfg["model_type"] == "seq2seq" and not keeps_zwnj(tokenizer)
    hconfig["candidates"] = words
    model = build_model(hconfig, tokenizer, cfg["model_id"])
    if cfg["gradient_checkpointing"]:
        (model if cfg["model_type"] == "seq2seq" else model.encoder).gradient_checkpointing_enable()
    seq2seq = cfg["model_type"] == "seq2seq"
    train_data, dev_data = make_features(hconfig, tokenizer, train_df), make_features(hconfig, tokenizer, dev_df)

    common = dict(
        output_dir=str(ckpt_dir / "trainer"), save_strategy="no", eval_strategy="epoch", logging_strategy="steps",
        logging_steps=10, report_to=["tensorboard"], logging_dir=str(run_dir / "tb"), disable_tqdm=True,
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
        per_device_eval_batch_size=cfg["batch_size"] * 2, learning_rate=cfg["lr"], weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"], lr_scheduler_type="linear", max_grad_norm=1.0, optim="adamw_torch",
        num_train_epochs=cfg["epochs"], max_steps=SMOKE_STEPS if smoke else -1, fp16=cfg["precision"] == "fp16",
        seed=cfg["seed"], data_seed=cfg["seed"], remove_unused_columns=False, label_names=["labels"],
    )
    monitor = Monitor(run_dir / "train_log.csv", cfg["patience"])
    if seq2seq:
        decoder = Predictor(model, tokenizer, hconfig, model.device, half=False)
        dev_items = list(zip(dev_df["sentence"], dev_df["start"], dev_df["end"], dev_df["spelling"]))

        def compute_metrics(eval_pred):
            ids = np.where(eval_pred.predictions != -100, eval_pred.predictions, tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
            preds = [decoder.from_output(t, item) for t, item in zip(texts, dev_items)]
            return {"accuracy": float(np.mean([p.aligned and p.pronunciation == g for p, g in zip(preds, dev_df["pronunciation"])]))}

        args = Seq2SeqTrainingArguments(**common, predict_with_generate=True, generation_max_length=cfg["max_length"], generation_num_beams=1)
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_data, eval_dataset=dev_data, processing_class=tokenizer,
                                 data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), compute_metrics=compute_metrics, callbacks=[monitor])
    else:
        def compute_metrics(eval_pred):
            return {"accuracy": float((eval_pred.predictions.argmax(-1) == eval_pred.label_ids).mean())}

        trainer = Trainer(model=model, args=TrainingArguments(**common), train_dataset=train_data, eval_dataset=dev_data,
                          data_collator=lambda f: collate(f, tokenizer.pad_token_id), compute_metrics=compute_metrics, callbacks=[monitor])

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    if monitor.nonfinite:
        raise NonFiniteLoss(f"non-finite loss in {cfg['precision']}")
    model.load_state_dict(monitor.best_state)
    save_checkpoint(model, tokenizer, hconfig, ckpt_dir)
    shutil.rmtree(ckpt_dir / "trainer", ignore_errors=True)
    return {
        "train_time_s": round(train_time, 1),
        "epochs_trained": round(trainer.state.epoch, 2),
        "best_epoch": monitor.best_epoch,
        "best_dev_accuracy_during_training": monitor.best,
        "steps": trainer.state.global_step,
        "peak_gpu_memory_train_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_gpu_memory_reserved_train_mb": round(torch.cuda.max_memory_reserved() / 2**20),
        "params": sum(p.numel() for p in model.parameters()),
        "checkpoint_size_mb": dir_size_mb(ckpt_dir),
    }


def fit_with_fallbacks(cfg: dict, train_df, dev_df, run_dir: Path, ckpt_dir: Path, smoke: bool) -> dict:
    """On OOM: halve the batch and double grad accumulation, then gradient checkpointing.
    On a non-finite loss in fp16: retry in fp32. Every change is recorded in cfg["adjustments"]."""
    while True:
        shutil.rmtree(run_dir / "tb", ignore_errors=True)
        try:
            return fit(cfg, train_df, dev_df, run_dir, ckpt_dir, smoke)
        except torch.cuda.OutOfMemoryError:
            if cfg["batch_size"] > 1:
                cfg["batch_size"] //= 2
                cfg["grad_accum"] *= 2
                change = f"OOM -> batch {cfg['batch_size']} x grad_accum {cfg['grad_accum']}"
            elif not cfg["gradient_checkpointing"]:
                cfg["gradient_checkpointing"] = True
                change = "OOM -> gradient checkpointing"
            else:
                raise
        except NonFiniteLoss:
            if cfg["precision"] != "fp16":
                raise
            cfg["precision"] = "fp32"
            change = "non-finite loss in fp16 -> fp32"
        print(f"[fallback] {change}", flush=True)
        cfg["adjustments"].append(change)
        gc.collect()
        torch.cuda.empty_cache()


def test_sets(cfg: dict) -> dict[str, pd.DataFrame]:
    _, _, test = SPLITS[cfg["split"]]
    sets = {test: load_split(test)}
    if REAL_TEST.exists():
        sets["real_test"] = label_rows(read_csv(REAL_TEST), load_words())
    return sets


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(cfg: dict, setting: str, seed: int, lr_search: float | None, smoke: bool) -> None:
    name = cfg["model_name"]
    rel = Path("lr_search", f"lr{lr_search:g}") if lr_search else Path(setting)
    run_dir = SMOKE_DIR / name if smoke else RUNS / name / rel / f"seed{seed}"
    ckpt_dir = Path("checkpoints/_smoke") / name if smoke else CHECKPOINT_RUNS / name / rel / f"seed{seed}"
    if (run_dir / "metrics.json").exists():
        print(f"skip {run_dir}: metrics.json exists")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    lr = lr_search or (cfg["lr_grid"][0] if smoke else best_lr(name))
    cfg = {"split": "main", "train_file": None, "output": None, **cfg, **SETTINGS[setting],
           "setting": setting, "seed": seed, "lr": lr, "adjustments": []}
    train_name, dev_name, _ = SPLITS[cfg["split"]]
    cfg["train_file"] = cfg["train_file"] or train_name
    train_df, dev_df = load_split(cfg["train_file"]), load_split(dev_name)
    if smoke:
        n = SMOKE_STEPS * cfg["batch_size"] * cfg["grad_accum"]
        train_df, dev_df = train_df.sample(n=min(n, len(train_df)), random_state=seed), dev_df.sample(n=64, random_state=seed)

    reuse = RUNS / name / "lr_search" / f"lr{lr:g}" / f"seed{seed}"
    reuse_ckpt = CHECKPOINT_RUNS / name / "lr_search" / f"lr{lr:g}" / f"seed{seed}"
    if setting == "main" and not lr_search and not smoke and (reuse / "metrics.json").exists() and reuse_ckpt.exists():
        # The lr-search run with the best lr is exactly the seed-42 main run, so it is reused, not retrained.
        shutil.copytree(reuse, run_dir, dirs_exist_ok=True)
        shutil.copytree(reuse_ckpt, ckpt_dir, dirs_exist_ok=True)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        (run_dir / "metrics.json").unlink()
        cfg = {**json.loads((run_dir / "config.json").read_text()), "setting": setting, "reused_from": str(reuse)}
    else:
        metrics = {"train": fit_with_fallbacks(cfg, train_df, dev_df, run_dir, ckpt_dir, smoke)}
        cfg["environment"] = environment()
        write_json(run_dir / "config.json", cfg)
        gc.collect()
        torch.cuda.empty_cache()
        predictor = Predictor.load(ckpt_dir)
        metrics["dev"], _ = evaluate(predictor, dev_df)

    if not lr_search:
        predictor = Predictor.load(ckpt_dir)
        tests = {"test": dev_df} if smoke else test_sets(cfg)
        for test_name, df in tests.items():
            metrics[test_name], predictions = evaluate(predictor, df)
            suffix = "" if test_name in ("test", "unseen_test") else f"_{test_name}"
            predictions.to_csv(run_dir / f"predictions{suffix}.csv", index=False, encoding="utf-8")
    write_json(run_dir / "metrics.json", metrics)
    if smoke or (setting not in KEEP_CHECKPOINT and not lr_search):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    summary = {k: round(v["accuracy"], 4) for k, v in metrics.items() if "accuracy" in v}
    print(f"done {run_dir}: {summary} adjustments={cfg['adjustments']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--setting", default="main", choices=list(SETTINGS))
    parser.add_argument("--lr-search", type=float, default=None, help="train seed 42 on the main split with this lr (dev only)")
    parser.add_argument("--smoke", action="store_true", help=f"{SMOKE_STEPS} steps on a tiny subset, results in {SMOKE_DIR}")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("training needs a GPU")
    if "T4" not in torch.cuda.get_device_name(0):
        print(f"warning: expected a Tesla T4, got {torch.cuda.get_device_name(0)}; set CUDA_VISIBLE_DEVICES", flush=True)
    run(load_config(args.config), args.setting, args.seed, args.lr_search, args.smoke)


if __name__ == "__main__":
    main()
