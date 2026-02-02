"""Render a pruning curve from the debug log."""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


DEFAULT_LOG_PATH = "pruning_debug.jsonl"
DEFAULT_OUTPUT_PATH = "pruning_curve.png"
DEFAULT_SHOW_PLOT = True


def read_pruning_log(
    log_path: str,
) -> Tuple[Optional[float], Optional[int], Optional[float], List[Tuple[int, float]]]:
    """Parse debug events from the pruning log."""
    baseline_score = None
    final_params = None
    compression = None
    points: List[Tuple[int, float]] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = event.get("event")
            if event_type == "baseline":
                baseline_score = event.get("baseline_score")
            if event_type == "final":
                final_params = event.get("final_params")
                compression = event.get("compression")
            if event_type == "accept":
                params = event.get("params")
                score = event.get("score_full")
                if params is not None and score is not None:
                    points.append((params, score))

    return baseline_score, final_params, compression, points


def build_curve_points(
    baseline_score: Optional[float],
    final_params: int,
    compression: float,
    points: List[Tuple[int, float]],
) -> Tuple[List[float], List[float]]:
    """Convert raw log stats into x/y coordinates for plotting."""
    initial_params = int(round(final_params / max(1.0 - compression, 1e-9)))
    xs: List[float] = []
    ys: List[float] = []

    if baseline_score is not None:
        xs.append(0.0)
        ys.append(float(baseline_score))

    for params, score in points:
        prune_pct = 100.0 * (1.0 - (params / initial_params))
        xs.append(prune_pct)
        ys.append(float(score))

    return xs, ys


def plot_pruning_curve(xs: List[float], ys: List[float], output_path: str) -> None:
    """Plot and save the pruning curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlabel("Pruning percentage (%)")
    plt.ylabel("Embedding similarity")
    plt.title("Pruning curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def main() -> None:
    """Entrypoint for pruning curve visualization."""
    baseline_score, final_params, compression, points = read_pruning_log(
        DEFAULT_LOG_PATH
    )
    if final_params is None or compression is None:
        raise ValueError("Missing final_params or compression in pruning_debug.jsonl.")

    xs, ys = build_curve_points(baseline_score, final_params, compression, points)
    plot_pruning_curve(xs, ys, DEFAULT_OUTPUT_PATH)

    if DEFAULT_SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
