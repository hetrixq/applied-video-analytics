from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np

from .utils import load_yaml, save_yaml, save_json, read_lines, set_global_seed, compute_metrics
from .data import load_annotations, extract_motion_scores, label_windows
from .model import fit_threshold, Threshold1D

def run_split(split_name: str, videos: list[str], cfg, annotations, model: Threshold1D | None):
    rows = []
    all_scores = []
    all_y = []

    overlap_thr = float(cfg["evaluation"]["label_overlap_frac"])

    for vrel in videos:
        vpath = os.path.join(cfg["data"]["raw_dir"], vrel)
        scores, windows = extract_motion_scores(vpath, cfg)
        y_true = label_windows(vrel, windows, annotations, overlap_thr)

        for (t0, t1), s, yt in zip(windows, scores.tolist(), y_true.tolist()):
            rows.append({
                "split": split_name,
                "video": vrel,
                "t0": float(t0),
                "t1": float(t1),
                "motion_score": float(s),
                "y_true": int(yt),
            })
        all_scores.append(scores)
        all_y.append(y_true)

    scores_all = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=float)
    y_all = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,), dtype=int)

    if model is None:
        model = fit_threshold(
            scores=scores_all,
            y_true=y_all,
            n_thresholds=int(cfg["model"]["search"]["n_thresholds"]),
            metric=str(cfg["model"]["search"]["metric"]),
        )

    y_pred = model.predict(scores_all)
    metrics = compute_metrics(y_all, y_pred)

    for i, r in enumerate(rows):
        r["y_pred"] = int(y_pred[i]) if i < len(y_pred) else 0

    return model, metrics, pd.DataFrame(rows)

def main(config_path: str):
    cfg = load_yaml(config_path)
    set_global_seed(int(cfg["project"]["random_seed"]))

    annotations = load_annotations(cfg["data"]["annotations_csv"])

    splits_dir = cfg["data"]["splits_dir"]
    train_v = read_lines(os.path.join(splits_dir, cfg["data"]["train_split"]))
    val_v   = read_lines(os.path.join(splits_dir, cfg["data"]["val_split"]))
    test_v  = read_lines(os.path.join(splits_dir, cfg["data"]["test_split"]))

    model, train_metrics, train_df = run_split("train", train_v, cfg, annotations, model=None)
    _, val_metrics, val_df   = run_split("val", val_v, cfg, annotations, model=model)
    _, test_metrics, test_df = run_split("test", test_v, cfg, annotations, model=model)

    out_dir = os.path.join(cfg["output"]["runs_dir"], cfg["output"]["run_name"])
    os.makedirs(out_dir, exist_ok=True)

    save_yaml(cfg, os.path.join(out_dir, "params.yaml"))
    save_json({"model_type": cfg["model"]["type"], "threshold": float(model.threshold)}, os.path.join(out_dir, "model.json"))
    save_json({"train": train_metrics, "val": val_metrics, "test": test_metrics}, os.path.join(out_dir, "metrics.json"))

    preds = pd.concat([train_df, val_df, test_df], ignore_index=True)
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    print("Saved run to:", out_dir)
    print("Learned threshold:", float(model.threshold))
    print("VAL:", val_metrics)
    print("TEST:", test_metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
