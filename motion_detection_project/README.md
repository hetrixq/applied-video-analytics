# Motion Detection System in Video Streams

This project is dedicated to the development and analysis of a motion detection system in video streams as part of the course *“Applied Video Analytics”*.

The project treats motion detection as an **engineering system**, including requirement formalization, a reproducible experimental protocol and analysis of baseline system behavior.

## Project Structure

```sh
.
├── PROJECT_PASSPORT.md     # Task and system requirements formalization
├── README.md               # General project description (this file)
│
├── configs/
│ └── baseline.yaml         # Baseline experiment configuration
│
├── data/
│ └── annotations.csv       # Motion interval annotations
│
├── splits/
│ ├── train.txt             # Training split (by video)
│ ├── val.txt               # Validation split
│ └── test.txt              # Test split
│
├── src/
│ └── baseline/             # Baseline pipeline implementation
│     ├── __init__.py
│     ├── baseline.py
│     ├── data.py
│     ├── model.py
│     ├── utils.py
│     ├── visualize.py
│     └── README.md         # Detailed baseline logic description
│
├── runs/                   # Baseline run results (generated after execution)
└── reports/                # Analytical reports
````

## Task Formalization

A detailed description of the application task, usage context, input and output data, as well as system constraints is provided in **[`PROJECT_PASSPORT.md`](PROJECT_PASSPORT.md)**.

## Baseline Solution

As part of the project, a **reproducible baseline pipeline** for motion detection has been implemented, including:
* a fixed experimental protocol, 
* a simple standard model,
* training and evaluation procedures,
* storage of experiment artifacts.

The baseline implementation is located in **[`src/baseline/`](src/baseline/)**.

A detailed description of the baseline logic, experimental protocol, and
saved artifacts is provided in **[`src/baseline/README.md`](src/baseline/README.md)**.
Detailed run instructions are also available there.

## Experimental Results

Results of each baseline run are stored in the **[`runs/`](runs/)** directory and include the experiment configuration, model parameters, evaluation metrics and prediction outputs.

## Analysis

An engineering analysis of the baseline behavior, its limitations and
characteristic error patterns is provided in **[`reports/`](reports/)**.
