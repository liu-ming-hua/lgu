## LGU — Local Graph Uncertainty

This repository contains scripts and analysis tools for generating model answers and computing graph consistency and uncertainty metrics (LGU). This README helps you get started, reproduce experiments, and inspect results locally.

**Key Features**
- Generate model answers: `run_generate_answers.sh`
- Compute LGU metrics: `compute_lgu.sh` / `compute_lgu.py`
- Analysis and visualization: Jupyter notebooks in `notebooks/` and analysis scripts under `result/`

**Repository Overview**
- `generate_answers.py` / `run_generate_answers.sh`: Batch inference scripts that save outputs to the `result/` directory.
- `compute_lgu.sh` / `compute_lgu.py`: Scripts to compute LGU and consistency metrics based on generated answers.
- `graph_consistency_verification/`: Tools for checking graph properties such as symmetry and transitivity (`symmetry.py`, `transitivity.py`).
- `notebooks/`: Interactive analysis and ablation study notebooks (e.g., `analyze_run.ipynb`, `ablation_study.ipynb`).
- `result/`: Model outputs and evaluation results organized by model and task; includes aggregated files such as under `10_answer/`.

## Quick Start

1. Change to the project directory:

```bash
cd LGU
```

2. Create and activate the Conda environment:

```bash
conda env update -f environment.yaml
conda activate lgu
```

3. Generate model answers (example):

```bash
./run_generate_answers.sh
```

The script will save inference outputs into model/task-specific subdirectories under `result/`.

4. Compute LGU metrics:

```bash
./compute_lgu.sh
```

Or run the Python script directly:

```bash
python3 semantic_uncertainty/compute_lgu.py --results_dir result --out_dir result/metrics
```

5. Analyze results and plot:

- Open the notebooks in `notebooks/` (for example, `analyze_run.ipynb`) for interactive visualization.
- The `result/10_answer/` directory contains summary CSV files (`auroc_results.csv`, `auarc_results.csv`, `ece_results.csv`) that can be used for tables and plots.

## Main Scripts

- `run_generate_answers.sh`: Wrapper for common inference commands (adjust parameters inside the script according to your hardware and models).
- `compute_lgu.sh`: Invokes `compute_lgu.py` or other analysis scripts to compute graph consistency and uncertainty metrics.
- `rename_folders.py`: Helper to normalize naming of model output directories under `result/`.
- `graph_consistency_verification/`: Contains `symmetry.py` and `transitivity.py` to check whether generated relation graphs satisfy properties like symmetry and transitivity.

## Outputs and Logs

- All model outputs and intermediate files are stored under `result/`, organized by model and task.
- Run-time logs are typically located in `logs/` folders within each model/task subdirectory.

## Recommendations and Notes

- Ensure you have enough disk space for model outputs and intermediate files.
- If using GPUs, verify device settings and batch sizes in the scripts to avoid out-of-memory errors.
- Some model directories (especially for large models) may contain compressed or sharded files—confirm paths in `run_generate_answers.sh` before running.


