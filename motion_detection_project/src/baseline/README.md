# Motion Detection Baseline

End-to-end baseline pipeline for **motion detection** in video streams.  
Problem formulation: **binary classification over temporal windows** ( `motion` / `no_motion` ).

## At a Glance

* **Input**: videos in `data/raw/` + temporal annotations in `data/annotations.csv`
* **Split**: fixed lists in `splits/*.txt` (split by **videos**, not frames)
* **Model**: threshold-based classifier on `motion_score`
* **Training**: grid search for the best threshold on the training set
* **Eval**: accuracy / precision / recall / F1 (+ TP / TN / FP / FN)
* **Outputs**: artifacts saved under `runs/<run_name>/`

## Quickstart

1. Put video files into `data/raw/` (subdirectories are OK).
2. Fill `splits/train.txt`,   `splits/val.txt`,  `splits/test.txt` with relative paths (one per line).
3. Create annotations in `data/annotations.csv`.
4. Install dependencies:
    

    ```bash
    # reproducible (recommended)
    pip install -r requirements.lock

    # minimal (for understanding / quick start)
    pip install -r requirements.txt
    ```

5. Run:
    
    ```bash
    python -m src.baseline --config configs/baseline.yaml
    ```

Artifacts are saved to `runs/<run_name>/` :

* `params.yaml` – configuration used for the run
* `model.json` – learned threshold value
* `metrics.json` – metrics on train / val / test splits
* `predictions.csv` – per-window predictions

## Annotations Format

Temporal motion intervals live in `data/annotations.csv` :

* `video` — path relative to `data/raw/` (must match entries in split files)
* `t_start_sec` — start time of motion interval (seconds)
* `t_end_sec` — end time of motion interval (seconds)

Example:

```csv
video,t_start_sec,t_end_sec
cam1/scene_001.mp4,3.2,8.9
cam1/scene_001.mp4,15.0,16.4
```

If a video contains no motion at all, it can be omitted from the CSV file
and will be treated as entirely negative ( `no_motion` ).

## Reproducibility

Reproducibility is ensured by:

* fixed split files
* a fixed configuration (FPS, resize, window size, thresholds, metrics)
* a fixed `random_seed`

## Environment

* **Python**: 3.9.x
* **OS**: macOS Sonoma, Apple Silicon (M2)
