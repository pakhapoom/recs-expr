import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_EXPERIMENTS = ["random", "popularity", "two_tower"]


def load_ndcg_metrics(metrics_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    ndcg = metrics[metrics["metric"].str.startswith("ndcg@")].copy()
    if ndcg.empty:
        raise ValueError(f"No ndcg@k rows found in {metrics_path}")

    ndcg["k"] = ndcg["metric"].str.split("@").str[-1].astype(int)
    ndcg = ndcg.sort_values("k")
    return ndcg[["k", "value"]]


def format_label(experiment: str) -> str:
    parts = experiment.replace("-", "_").split("_")
    formatted = [part.upper() if len(part) <= 3 else part.title() for part in parts]
    return "-".join(formatted)


def plot_ndcg(metrics_dir: Path, experiments: list[str], output_path: Path, title: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    all_k_values = set()

    for experiment in experiments:
        metrics_path = metrics_dir / f"{experiment}.csv"
        ndcg = load_ndcg_metrics(metrics_path)
        all_k_values.update(ndcg["k"].tolist())
        ax.plot(
            ndcg["k"],
            ndcg["value"],
            marker="o",
            linewidth=2,
            label=format_label(experiment),
        )

    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("NDCG@k")
    ax.set_xticks(sorted(all_k_values))
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="data/metrics")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output_path", default="data/metrics/ndcg_at_k.png")
    parser.add_argument("--title", default="NDCG@k Comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_ndcg(
        metrics_dir=Path(args.metrics_dir),
        experiments=args.experiments,
        output_path=Path(args.output_path),
        title=args.title,
    )
    print(f"Saved plot → {args.output_path}")