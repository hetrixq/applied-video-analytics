from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]


def load_metrics(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "predictions.csv")
    return pd.read_csv(path)


def _confusion_counts(df: pd.DataFrame) -> Tuple[int, int, int, int]:
    """Return tp, tn, fp, fn for dataframe that has y_true and y_pred."""
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def plot_metrics_bar(metrics: Dict[str, Any], out_path: str | None = None) -> None:
    splits = [s for s in ["train", "val", "test"] if s in metrics]
    if not splits:
        print("No splits found in metrics.json")
        return

    values = np.array(
        [[metrics[s].get(k, np.nan) for k in METRIC_KEYS] for s in splits], dtype=float
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.arange(len(METRIC_KEYS))
    width = 0.25 if len(splits) >= 3 else 0.35

    for i, s in enumerate(splits):
        ax.bar(x + (i - (len(splits) - 1) / 2) * width, values[i], width, label=s)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_KEYS)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Metrics by split")
    ax.legend()

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_confusion_matrices(preds: pd.DataFrame, out_path: str | None = None) -> None:
    splits = preds["split"].unique().tolist()
    splits = [s for s in ["train", "val", "test"] if s in splits]

    n = len(splits)
    if n == 0:
        print("No split column or empty predictions.")
        return

    # One figure, one axis per split
    fig = plt.figure(figsize=(4 * n, 4))
    for i, s in enumerate(splits, start=1):
        ax = fig.add_subplot(1, n, i)
        df = preds[preds["split"] == s]
        tp, tn, fp, fn = _confusion_counts(df)
        mat = np.array([[tn, fp], [fn, tp]], dtype=int)

        im = ax.imshow(mat)
        ax.set_title(f"Confusion ({s})")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])

        for (r, c), v in np.ndenumerate(mat):
            ax.text(c, r, str(v), ha="center", va="center")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_score_distributions(preds: pd.DataFrame, out_path: str | None = None) -> None:
    splits = preds["split"].unique().tolist()
    splits = [s for s in ["train", "val", "test"] if s in splits]

    if not splits:
        print("No splits found in predictions.")
        return

    fig = plt.figure(figsize=(5 * len(splits), 4))
    for i, s in enumerate(splits, start=1):
        ax = fig.add_subplot(1, len(splits), i)
        df = preds[preds["split"] == s].copy()

        # Guard: some splits might have only one class
        for label in [0, 1]:
            dd = df[df["y_true"] == label]["motion_score"].astype(float).to_numpy()
            if len(dd) == 0:
                continue
            ax.hist(dd, bins=30, alpha=0.6, label=f"y_true={label}")

        ax.set_title(f"motion_score dist ({s})")
        ax.set_xlabel("motion_score")
        ax.set_ylabel("count")
        ax.legend()

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_timeline_for_video(
    preds: pd.DataFrame,
    video: str,
    split: str | None = None,
    out_path: str | None = None,
) -> None:
    df = preds[preds["video"] == video].copy()
    if split is not None:
        df = df[df["split"] == split].copy()

    if df.empty:
        msg = f"No rows found for video='{video}'"
        if split:
            msg += f" in split='{split}'"
        print(msg)
        return

    df = df.sort_values(["t0"]).reset_index(drop=True)

    # Timeline values
    t = df["t0"].to_numpy()
    score = df["motion_score"].astype(float).to_numpy()
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(t, score, linewidth=1.5, label="motion_score")
    ax.step(t, y_true, where="post", label="y_true")
    ax.step(t, y_pred, where="post", label="y_pred")

    ax.set_title(f"Timeline: {video}" + (f" ({split})" if split else ""))
    ax.set_xlabel("time (sec)")
    ax.legend()
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def ensure_out_dir(run_dir: str, out_dir: str | None) -> str:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    # default: create runs/<run_name>/plots
    out_dir2 = os.path.join(run_dir, "plots")
    os.makedirs(out_dir2, exist_ok=True)
    return out_dir2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path like runs/baseline_001")
    ap.add_argument(
        "--save",
        action="store_true",
        help="Save plots to run_dir/plots instead of showing",
    )
    ap.add_argument(
        "--out_dir", default=None, help="Optional output directory for saved plots"
    )
    ap.add_argument(
        "--video",
        default=None,
        help="Optional: plot timeline for this video (e.g., vid6.mp4)",
    )
    ap.add_argument(
        "--split",
        default=None,
        help="Optional: restrict timeline to split (train/val/test)",
    )
    args = ap.parse_args()

    metrics = load_metrics(args.run_dir)
    preds = load_predictions(args.run_dir)

    if args.save:
        out_dir = ensure_out_dir(args.run_dir, args.out_dir)
        plot_metrics_bar(metrics, os.path.join(out_dir, "metrics_by_split.png"))
        plot_confusion_matrices(preds, os.path.join(out_dir, "confusion_by_split.png"))
        plot_score_distributions(
            preds, os.path.join(out_dir, "score_distributions.png")
        )

        if args.video:
            plot_timeline_for_video(
                preds,
                video=args.video,
                split=args.split,
                out_path=os.path.join(
                    out_dir, f"timeline_{args.video.replace('/', '_')}.png"
                ),
            )
        print(f"Saved plots to: {out_dir}")
    else:
        plot_metrics_bar(metrics, out_path=None)
        plot_confusion_matrices(preds, out_path=None)
        plot_score_distributions(preds, out_path=None)
        if args.video:
            plot_timeline_for_video(
                preds, video=args.video, split=args.split, out_path=None
            )


if __name__ == "__main__":
    main()
