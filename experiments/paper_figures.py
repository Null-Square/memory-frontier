"""Generate the four main manuscript figures.

This script is intentionally outside CI. It uses only frozen theorem/census values
and elementary asymptotic formulas so that the paper figures are reproducible
without rerunning the exhaustive research programs.

Usage:
    python experiments/paper_figures.py --outdir paper/generated_figures

Requires matplotlib in addition to the project dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError as exc:  # pragma: no cover - outside CI
        raise SystemExit(
            "paper_figures.py requires matplotlib; install it before running"
        ) from exc
    return plt, FancyArrowPatch, FancyBboxPatch


def save_forward_equivalence(outdir: Path) -> None:
    """Draw three forward-equivalent bases with different dormant scaffolds."""
    plt, FancyArrowPatch, FancyBboxPatch = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.75), constrained_layout=True)
    orders = [1, 2, 4]
    xs = np.asarray([0.08, 0.29, 0.50, 0.71, 0.92])
    y = 0.50
    box_w = 0.12
    box_h = 0.20

    for ax, order in zip(axes, orders):
        ax.axis("off")

        for i, x in enumerate(xs):
            face = "white" if i == 0 else "0.92"
            box = FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.02",
                facecolor=face,
                edgecolor="black",
                linewidth=1.2,
                transform=ax.transAxes,
            )
            ax.add_patch(box)
            ax.text(x, y, f"m{i}", ha="center", va="center", fontsize=9, transform=ax.transAxes)

        for i in range(4):
            learned = i < order
            arrow = FancyArrowPatch(
                (xs[i] + box_w / 2 + 0.01, y),
                (xs[i + 1] - box_w / 2 - 0.01, y),
                arrowstyle="->",
                mutation_scale=10,
                linestyle="--" if learned else "-",
                linewidth=1.7 if learned else 2.2,
                transform=ax.transAxes,
            )
            ax.add_patch(arrow)
            ax.text(
                (xs[i] + xs[i + 1]) / 2,
                y + 0.14,
                "learned" if learned else "prewired",
                ha="center",
                va="bottom",
                fontsize=7,
                transform=ax.transAxes,
            )

        ax.text(xs[0], 0.25, "forward-active", ha="center", fontsize=7.5, transform=ax.transAxes)
        ax.text(xs[-1], 0.25, "predictive readout", ha="center", fontsize=7.5, transform=ax.transAxes)
        ax.set_title(f"same current predictor, order {order}", fontsize=10, pad=2)

    fig.suptitle(
        "Forward-equivalent bases can expose different construction orders",
        fontsize=11.5,
    )
    fig.text(
        0.5,
        0.01,
        "white = forward-active at the base; gray = dormant; dashed = learned edge; solid = dormant prewire",
        ha="center",
        fontsize=8,
    )
    fig.savefig(outdir / "figure1_forward_equivalence.pdf", bbox_inches="tight")
    fig.savefig(outdir / "figure1_forward_equivalence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_order_hierarchy(outdir: Path) -> None:
    """Draw the support/operator/loss order hierarchy and cancellation gaps."""
    plt, FancyArrowPatch, FancyBboxPatch = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 3.0), constrained_layout=True)
    ax.axis("off")

    centers = [(0.17, 0.61), (0.50, 0.61), (0.83, 0.61)]
    labels = [
        (r"$d_{\rm support}$", "source-valid path cost"),
        (r"$d_{\rm operator}$", "quotient occupancy order"),
        (r"$d_{\rm loss}$", "scalar visibility order"),
    ]
    for (cx, cy), (symbol, subtitle) in zip(centers, labels):
        box = FancyBboxPatch(
            (cx - 0.115, cy - 0.135),
            0.23,
            0.27,
            boxstyle="round,pad=0.02",
            facecolor="white",
            edgecolor="black",
            linewidth=1.25,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.035, symbol, ha="center", va="center", fontsize=15, transform=ax.transAxes)
        ax.text(cx, cy - 0.055, subtitle, ha="center", va="center", fontsize=8, transform=ax.transAxes)

    for left, right, text in [
        (centers[0], centers[1], "path/operator\ncancellation"),
        (centers[1], centers[2], "decoder\ncancellation"),
    ]:
        arrow_y = 0.61
        arrow = FancyArrowPatch(
            (left[0] + 0.122, arrow_y),
            (right[0] - 0.122, arrow_y),
            arrowstyle="->",
            mutation_scale=13,
            linewidth=1.35,
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)
        ax.text(
            (left[0] + right[0]) / 2,
            0.39,
            text,
            ha="center",
            va="top",
            fontsize=7.8,
            transform=ax.transAxes,
        )

    ax.text(
        0.5,
        0.19,
        r"$d_{\rm support}\leq d_{\rm operator}\leq d_{\rm loss}$",
        ha="center",
        va="center",
        fontsize=16,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.055,
        "Generic continuous choices saturate the hierarchy; exact cancellations can make either inequality strict.",
        ha="center",
        va="center",
        fontsize=8.2,
        transform=ax.transAxes,
    )
    fig.savefig(outdir / "figure2_order_hierarchy.pdf", bbox_inches="tight")
    fig.savefig(outdir / "figure2_order_hierarchy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_census(outdir: Path) -> None:
    """Plot the frozen 1,000-controller same-forward-class census."""
    plt, _, _ = _plt()
    orders = np.arange(1, 6)
    counts = np.asarray([235, 282, 244, 155, 84])

    fig, ax = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    bars = ax.bar(orders, counts)
    ax.set_xlabel(r"exact order  $d_{\rm support}=d_{\rm operator}=d_{\rm loss}$")
    ax.set_ylabel("controllers")
    ax.set_title("1,000 controllers in one exact forward-equivalence class")
    ax.set_xticks(orders)
    ax.set_ylim(0, 320)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 6, str(int(count)), ha="center", fontsize=9)
    ax.text(
        0.02,
        0.96,
        "only dormant zero-cost wiring varies",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.savefig(outdir / "figure3_forward_class_census.pdf", bbox_inches="tight")
    fig.savefig(outdir / "figure3_forward_class_census.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def probability_time(degree: int, delta: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    if degree == 1:
        return threshold - delta
    if degree == 2:
        return np.log(threshold / delta)
    return (delta ** (2 - degree) - threshold ** (2 - degree)) / (degree - 2)


def softmax_leading_time(degree: int, delta: np.ndarray) -> np.ndarray:
    return delta ** (-degree) / degree


def save_geometry_time_map(outdir: Path) -> None:
    """Contrast two exact order-to-time maps without implying metric invariance."""
    plt, _, _ = _plt()
    deltas = np.logspace(-3.0, -1.35, 180)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.65), constrained_layout=True)
    for degree in range(1, 6):
        axes[0].loglog(deltas, probability_time(degree, deltas), label=f"d={degree}")
        axes[1].loglog(deltas, softmax_leading_time(degree, deltas), label=f"d={degree}")

    axes[0].set_title("Euclidean affine-probability flow")
    axes[1].set_title("rare-edge Euclidean softmax flow")
    for ax in axes:
        ax.set_xlabel(r"initial edge scale  $\delta$")
        ax.set_ylabel("leading completion time")
        ax.grid(True, which="both", linewidth=0.35)

    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(
        0.04,
        0.05,
        r"$d=1$: finite;  $d=2$: logarithmic;  $d\geq3$: $\Theta(\delta^{-(d-2)})$",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
    )
    axes[1].text(
        0.04,
        0.96,
        r"$\tau_d\sim \delta^{-d}/d$",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.suptitle(
        "Construction order is structural; the order-to-time map depends on optimization geometry",
        fontsize=11.2,
    )
    fig.savefig(outdir / "figure4_order_to_time_geometry.pdf", bbox_inches="tight")
    fig.savefig(outdir / "figure4_order_to_time_geometry.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("paper/generated_figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    save_forward_equivalence(args.outdir)
    save_order_hierarchy(args.outdir)
    save_census(args.outdir)
    save_geometry_time_map(args.outdir)


if __name__ == "__main__":
    main()
