"""Collects all runs into results/ (tables + plots) and refreshes the results tables in README.md and EXPERIMENT_LOG.md."""
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

RUNS, RESULTS, PLOTS = Path("runs"), Path("results"), Path("results/plots")
# Table order and the approximate sizes from the plan (actual parameter counts replace them once a model has run).
MODELS = [
    ("banglabert_small", "~13M"), ("banglabert", "~110M"), ("bangla_bert_base", "~165M"), ("muril_base", "~237M"),
    ("banglat5", "~250M"), ("xlm_roberta_base", "~278M"), ("indicbertv2_mlm", "~278M"), ("mdeberta_v3_base", "~280M"),
    ("mt5_small", "~300M"), ("byt5_small", "~300M"), ("banglabert_large", "~335M"), ("xlm_roberta_large", "~560M"),
    ("mt5_base", "~580M"),
]
LEARNING_CURVE = {"lc2": 2, "lc5": 5, "lc10": 10, "main": 14}
# latency file name -> variant label (runs/{model}/latency.json uses the model type)
LATENCY_VARIANTS = {"latency_cand": "candidate", "latency_target": "target-only"}
ABLATIONS = {
    "marker": [("main", "@ target @"), ("abl_marker_special", "<h> target </h> (new tokens)"), ("abl_marker_none", "no marker")],
    "encoder head": [("main", "opening marker token"), ("abl_head_mean", "mean of target tokens"), ("abl_head_cls", "[CLS]")],
    "seq2seq output": [("main", "full sentence"), ("abl_output_target", "target pronunciation only")],
}
# Reference categorical palette, first three slots (validated all-pairs for scatter plots).
COLORS = {"encoder": "#2a78d6", "seq2seq": "#eb6834", "candidate": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BENGALI_FONT = "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def pm(mean: float, std: float, scale: float = 100, digits: int = 2) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean * scale:.{digits}f}" if np.isnan(std) else f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def collect_runs() -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(RUNS.glob("*/*/seed*/metrics.json")):
        model, setting, seed = metrics_path.parts[1], metrics_path.parts[2], metrics_path.parts[3]
        metrics, config = read_json(metrics_path), read_json(metrics_path.parent / "config.json")
        for test_set, m in metrics.items():
            if test_set in ("train", "dev"):
                continue
            rows.append({
                "model": model, "setting": setting, "seed": int(seed[4:]), "test_set": test_set,
                "model_type": config["model_type"], "precision": config["precision"],
                "batch": f"{config['batch_size']}x{config['grad_accum']}", "lr": config["lr"],
                "adjustments": "; ".join(config["adjustments"]),
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "acc_2_readings": m["accuracy_by_readings"].get("2"), "acc_3_readings": m["accuracy_by_readings"].get("3"),
                "other_word_change_rate": m.get("other_word_change_rate"), "avg_generated_tokens": m.get("avg_generated_tokens"),
                "dev_accuracy": metrics["dev"]["accuracy"], **metrics["train"],
            })
    return pd.DataFrame(rows)


def latency_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(RUNS.glob("*/latency*.json")):
        lat, model = read_json(path), path.parts[1]
        variant = LATENCY_VARIANTS.get(path.stem) or lat["model_type"]
        for device in ("gpu", "cpu_4_threads", "cpu_1_thread"):
            r = lat[device]
            base = {"model": model, "variant": variant, "device": device, "device_name": r["device"],
                    "precision": r["precision"], "threads": r["threads"], "date": lat["date"],
                    "torch": lat["packages"]["torch"]}
            rows.append({**base, "batch_size": 1, "timed_calls": r["timed_calls"],
                         **{f"{k}_ms": v for k, v in r["batch1_full_ms"].items()},
                         "forward_p50_ms": r["batch1_forward_ms"]["p50"], "throughput_sps": 1000 / r["batch1_full_ms"]["mean"],
                         "phoneticize_p50_ms": r["phoneticize"]["total_ms"]["p50"], "ms_per_heteronym": r["phoneticize"]["ms_per_heteronym"],
                         "peak_gpu_memory_mb": r.get("peak_gpu_memory_mb"), "avg_generated_tokens": r.get("avg_generated_tokens")})
            for size, sps in r["throughput_sentences_per_s"].items():
                rows.append({**base, "batch_size": int(size), "timed_calls": r["timed_calls"], "throughput_sps": sps})
    return pd.DataFrame(rows)


def p50(latency: pd.DataFrame, model: str, variant: str, device: str) -> float:
    if latency.empty:
        return float("nan")
    hit = latency[(latency["model"] == model) & (latency["variant"] == variant)
                  & (latency["device"] == device) & (latency["batch_size"] == 1)]
    return float(hit["p50_ms"].iloc[0]) if len(hit) else float("nan")


def summarize(runs: pd.DataFrame, latency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, setting, test_set), g in runs.groupby(["model", "setting", "test_set"], sort=False):
        acc, f1 = mean_std(list(g["accuracy"])), mean_std(list(g["macro_f1"]))
        variant = ("candidate" if setting.startswith("cand_") else
                   "target-only" if setting.startswith("abl_output_target") else g["model_type"].iloc[0])
        rows.append({
            "model": model, "setting": setting, "test_set": test_set, "model_type": g["model_type"].iloc[0],
            "seeds": len(g), "accuracy_mean": acc[0], "accuracy_std": acc[1], "macro_f1_mean": f1[0], "macro_f1_std": f1[1],
            "acc_2_readings": g["acc_2_readings"].mean(), "acc_3_readings": g["acc_3_readings"].astype(float).mean(),
            "other_word_change_rate": g["other_word_change_rate"].astype(float).mean(),
            "params": int(g["params"].iloc[0]), "precision": "/".join(sorted(set(g["precision"]))), "batch": g["batch"].iloc[0],
            "lr": g["lr"].iloc[0], "train_time_s": g["train_time_s"].mean(), "epochs_trained": g["epochs_trained"].mean(),
            "best_epoch": g["best_epoch"].mean(), "peak_gpu_memory_train_mb": g["peak_gpu_memory_train_mb"].max(),
            "checkpoint_size_mb": g["checkpoint_size_mb"].iloc[0],
            "gpu_p50_ms": p50(latency, model, variant, "gpu"), "cpu4_p50_ms": p50(latency, model, variant, "cpu_4_threads"),
            "cpu1_p50_ms": p50(latency, model, variant, "cpu_1_thread"),
            "adjustments": "; ".join(sorted({a for a in g["adjustments"] if a})),
        })
    order = {m: i for i, (m, _) in enumerate(MODELS)}
    return pd.DataFrame(rows).sort_values(["setting", "test_set", "model"], key=lambda s: s.map(order) if s.name == "model" else s)


def fmt_params(n: float) -> str:
    return f"{n / 1e6:.1f}M"


def fmt_ms(x: float) -> str:
    return "" if np.isnan(x) else f"{x:.1f}"


def markdown_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summary_markdown(summary: pd.DataFrame) -> str:
    parts = ["# Summary\n", "Accuracy and macro-F1 in %, mean ± std over seeds. Latency: batch 1, full path, p50 ms "
             "(GPU: Tesla T4 at training precision; CPU: fp32, 4 / 1 threads).\n"]
    for (setting, test_set), g in summary.groupby(["setting", "test_set"], sort=False):
        table = pd.DataFrame({
            "model": g["model"], "type": g["model_type"], "params": g["params"].map(fmt_params), "precision": g["precision"],
            "batch": g["batch"], "lr": g["lr"].map(lambda x: f"{x:g}"), "seeds": g["seeds"],
            "accuracy": [pm(m, s) for m, s in zip(g["accuracy_mean"], g["accuracy_std"])],
            "macro-F1": [pm(m, s) for m, s in zip(g["macro_f1_mean"], g["macro_f1_std"])],
            "acc 2 / 3 readings": [f"{a * 100:.1f} / {b * 100:.1f}" if not np.isnan(b) else f"{a * 100:.1f} / -" for a, b in zip(g["acc_2_readings"], g["acc_3_readings"])],
            "other-word change": g["other_word_change_rate"].map(lambda x: "" if np.isnan(x) else f"{x * 100:.1f}"),
            "train time (min)": g["train_time_s"].map(lambda x: f"{x / 60:.1f}"), "epochs": g["epochs_trained"].map(lambda x: f"{x:.1f}"),
            "GPU p50": g["gpu_p50_ms"].map(fmt_ms), "CPU4 p50": g["cpu4_p50_ms"].map(fmt_ms), "CPU1 p50": g["cpu1_p50_ms"].map(fmt_ms),
        })
        parts.append(f"## {setting} / {test_set}\n\n{markdown_table(table)}\n")
    return "\n".join(parts)


def per_word() -> pd.DataFrame:
    frames = []
    for path in sorted(RUNS.glob("*/main/seed*/predictions.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        g = df.groupby(["spelling", "n_readings"])["correct"].agg(["mean", "size"]).reset_index()
        g["model"], g["seed"] = path.parts[1], path.parts[3]
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    out = df.groupby(["model", "spelling", "n_readings"]).agg(accuracy=("mean", "mean"), test_sentences=("size", "first"), seeds=("seed", "nunique"))
    return out.reset_index()


def learning_curve(runs: pd.DataFrame) -> pd.DataFrame:
    lc = runs[runs["setting"].isin(LEARNING_CURVE) & (runs["test_set"] == "test")]
    models_with_curve = set(lc[lc["setting"] != "main"]["model"])
    lc = lc[lc["model"].isin(models_with_curve)]
    rows = []
    for (model, setting), g in lc.groupby(["model", "setting"]):
        acc = mean_std(list(g["accuracy"]))
        rows.append({"model": model, "sentences_per_pronunciation": LEARNING_CURVE[setting], "train_sentences": LEARNING_CURVE[setting] * 368,
                     "accuracy_mean": acc[0], "accuracy_std": acc[1], "seeds": len(g)})
    return pd.DataFrame(rows).sort_values(["model", "sentences_per_pronunciation"]) if rows else pd.DataFrame()


def ablations_markdown(summary: pd.DataFrame) -> str:
    main = summary[summary["test_set"] == "test"]
    parts = ["# Ablations\n", "Main split test set, accuracy / macro-F1 in %, mean ± std over seeds.\n"]
    for group, variants in ABLATIONS.items():
        for model in sorted(set(main[main["setting"].isin([s for s, _ in variants[1:]])]["model"])):
            rows = []
            for setting, label in variants:
                hit = main[(main["model"] == model) & (main["setting"] == setting)]
                if len(hit):
                    h = hit.iloc[0]
                    rows.append({"variant": label, "setting": setting, "seeds": h["seeds"],
                                 "accuracy": pm(h["accuracy_mean"], h["accuracy_std"]), "macro-F1": pm(h["macro_f1_mean"], h["macro_f1_std"])})
            parts.append(f"## {group}: {model}\n\n{markdown_table(pd.DataFrame(rows))}\n")
    return "\n".join(parts)


def lr_search() -> pd.DataFrame:
    rows = []
    for path in sorted(RUNS.glob("*/best_lr.json")):
        best = read_json(path)
        for lr, r in best["search"].items():
            rows.append({"model": path.parts[1], "lr": float(lr), **r, "best": float(lr) == best["lr"]})
    return pd.DataFrame(rows)


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def legend(ax, kinds: set[str]) -> None:
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=COLORS[k], label=k) for k in COLORS if k in kinds]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2)


def scatter(points: pd.DataFrame, x: str, xlabel: str, title: str, out: Path) -> None:
    points = points.dropna(subset=[x])
    if points.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    for kind, g in points.groupby("kind"):
        ax.errorbar(g[x], g["accuracy_mean"] * 100, yerr=g["accuracy_std"].fillna(0) * 100, fmt="o", markersize=7,
                    color=COLORS[kind], ecolor=COLORS[kind], elinewidth=1, capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for row in points.itertuples():
        ax.annotate(row.label, (getattr(row, x), row.accuracy_mean * 100), xytext=(5, 4), textcoords="offset points", fontsize=7, color=INK_2)
    ax.set_xscale("log")
    style(ax, xlabel, "test accuracy (%)", title)
    legend(ax, set(points["kind"]))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plots(summary: pd.DataFrame, curve: pd.DataFrame, words: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    main = summary[(summary["test_set"] == "test") & summary["setting"].isin(["main", "cand_main"])].copy()
    main["kind"] = np.where(main["setting"] == "cand_main", "candidate", main["model_type"])
    main["label"] = np.where(main["setting"] == "cand_main", main["model"] + " (cand)", main["model"])
    scatter(main, "params", "parameters (log scale)", "Accuracy vs. model size", PLOTS / "accuracy_vs_params.png")
    scatter(main, "gpu_p50_ms", "GPU p50 latency, batch 1, ms (log scale)", "Accuracy vs. GPU latency (Tesla T4)", PLOTS / "accuracy_vs_latency_gpu.png")
    scatter(main, "cpu4_p50_ms", "CPU p50 latency, batch 1, 4 threads, ms (log scale)", "Accuracy vs. CPU latency", PLOTS / "accuracy_vs_latency_cpu.png")

    if not curve.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
        for color, (model, g) in zip(palette, curve.groupby("model")):
            ax.errorbar(g["sentences_per_pronunciation"], g["accuracy_mean"] * 100, yerr=g["accuracy_std"].fillna(0) * 100,
                        color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, label=model)
        ax.set_xticks(list(LEARNING_CURVE.values()))
        style(ax, "training sentences per pronunciation", "test accuracy (%)", "Learning curves")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        fig.tight_layout()
        fig.savefig(PLOTS / "learning_curves.png", dpi=200)
        plt.close(fig)

    if not words.empty:
        hardest = words.groupby("spelling")["accuracy"].mean().sort_values().head(20)[::-1]
        font = font_manager.FontProperties(fname=BENGALI_FONT) if Path(BENGALI_FONT).exists() else None
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
        ax.barh(range(len(hardest)), hardest.values * 100, color=COLORS["encoder"], height=0.7)
        ax.set_yticks(range(len(hardest)), hardest.index, fontproperties=font, fontsize=10)
        style(ax, "test accuracy (%), mean over models and seeds", "", f"Hardest words ({words['model'].nunique()} models, main split)")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(PLOTS / "hardest_words.png", dpi=200)
        plt.close(fig)


def status(model: str) -> str:
    if (RUNS / model / "latency.json").exists() and (RUNS / model / "main" / "kept_checkpoint.json").exists():
        return "done"
    return "running" if (RUNS / model).exists() else "pending"


def results_table(summary: pd.DataFrame, with_status: bool) -> str:
    main = summary[(summary["setting"] == "main") & (summary["test_set"] == "test")].set_index("model")
    rows = []
    for i, (model, size) in enumerate(MODELS, 1):
        state = status(model)
        if not with_status and state != "done":
            continue
        r = main.loc[model] if model in main.index else None
        best = RUNS / model / "best_lr.json"
        ckpt = RUNS / model / "main" / "kept_checkpoint.json"
        row = {"#": i, "model": model, "size": fmt_params(r["params"]) if r is not None else size,
               "precision": r["precision"] if r is not None else "", "best lr": f"{read_json(best)['lr']:g}" if best.exists() else "",
               "test acc (%)": pm(r["accuracy_mean"], r["accuracy_std"]) if r is not None else "",
               "macro-F1 (%)": pm(r["macro_f1_mean"], r["macro_f1_std"]) if r is not None else "",
               "train time / seed (min)": f"{r['train_time_s'] / 60:.1f}" if r is not None else "",
               "GPU p50 (ms)": fmt_ms(r["gpu_p50_ms"]) if r is not None else "", "CPU p50 (ms, 4 thr)": fmt_ms(r["cpu4_p50_ms"]) if r is not None else ""}
        if with_status:
            row["checkpoint"] = f"{read_json(ckpt)['size_mb']:.0f} MB" if ckpt.exists() else ""
            row["status"] = state
        rows.append(row)
    if not rows:
        return "No finished models yet."
    return markdown_table(pd.DataFrame(rows))


def replace_block(path: Path, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"(<!-- results-table -->\n).*?(\n<!-- /results-table -->)", lambda m: m.group(1) + content + m.group(2), text, flags=re.S)
    path.write_text(new, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    runs, latency = collect_runs(), latency_rows()
    latency.to_csv(RESULTS / "latency.csv", index=False)
    if runs.empty:
        print("no finished runs")
        return
    summary = summarize(runs, latency)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    (RESULTS / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    runs.to_csv(RESULTS / "runs.csv", index=False)
    words = per_word()
    words.to_csv(RESULTS / "per_word.csv", index=False)
    curve = learning_curve(runs)
    curve.to_csv(RESULTS / "learning_curve.csv", index=False)
    (RESULTS / "ablations.md").write_text(ablations_markdown(summary), encoding="utf-8")
    lr_search().to_csv(RESULTS / "lr_search.csv", index=False)
    plots(summary, curve, words)

    replace_block(Path("README.md"), results_table(summary, with_status=False))
    disk = subprocess.run(["du", "-sh", "checkpoints"], capture_output=True, text=True).stdout.split("\t")[0] if Path("checkpoints").exists() else "0"
    replace_block(Path("EXPERIMENT_LOG.md"), results_table(summary, with_status=True) + f"\n\nDisk used by `checkpoints/`: {disk}.")
    print(summary[(summary["setting"] == "main")][["model", "test_set", "seeds", "accuracy_mean", "accuracy_std", "gpu_p50_ms", "cpu4_p50_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
