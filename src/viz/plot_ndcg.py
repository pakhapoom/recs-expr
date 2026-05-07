import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_EXPERIMENTS = [
    "random",
    "popularity",
    "two_tower",
    "two_tower_v2",
    "two_tower_yt_cand_gen",
    "two_tower_yt_expand_dim",
]

EXPERIMENT_LABELS = {
    "two_tower": "two tower (manual)",
    "two_tower_v2": "two tower (single stage)",
    "two_tower_yt_cand_gen": "two tower (cand)",
    "two_tower_yt_expand_dim": "two tower (expand dim)",
}


def load_ndcg_at_k(metrics_path: Path, k: int = 10) -> float:
    metrics = pd.read_csv(metrics_path)
    metric_name = f"ndcg@{k}"
    values = metrics.loc[metrics["metric"].eq(metric_name), "value"]
    if values.empty:
        raise ValueError(f"No {metric_name} row found in {metrics_path}")

    return float(values.iloc[0])


def format_label(experiment: str) -> str:
    return EXPERIMENT_LABELS.get(experiment, experiment)


def plot_ndcg_at_k(
    k: int = 10,
    metrics_dir: Path = Path("data/metrics"),
    experiments: list[str] | None = None,
    output_path: Path = Path("data/viz/ndcg_at_k.png"),
    title: str | None = None,
) -> None:
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    experiments = experiments or DEFAULT_EXPERIMENTS
    labels = [format_label(experiment) for experiment in experiments]
    values = [
        load_ndcg_at_k(metrics_dir / f"{experiment}.csv", k=k)
        for experiment in experiments
    ]

    bars = ax.bar(labels, values, color="#3f6f9f", width=0.62)
    ax.set_title(title or f"NDCG@{k} Comparison")
    ax.set_ylabel(f"NDCG@{k}")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d0d0")
    ax.spines["bottom"].set_color("#d0d0d0")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", color="#d0d0d0")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="data/metrics")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output_path", default="data/viz/ndcg_at_k.png")
    parser.add_argument("--title")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_ndcg_at_k(
        k=args.k,
        metrics_dir=Path(args.metrics_dir),
        experiments=args.experiments,
        output_path=Path(args.output_path),
        title=args.title,
    )
    print(f"Saved plot → {args.output_path}")
