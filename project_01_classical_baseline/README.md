# Project 01 — Classical Clickbait Baseline

This checkpoint is a reproducible binary text-classification baseline using TF-IDF and Logistic Regression.

## Labels

- `0`: factual
- `1`: clickbait

## Final checkpoint notebook

Run `notebooks/07_project_01_checkpoint.ipynb` from top to bottom.

The notebook:

1. records package versions and the random seed;
2. loads and validates `data/practice_clickbait.csv`;
3. creates stratified train, validation, and test splits;
4. compares word and character TF-IDF configurations;
5. selects a configuration using validation Macro-F1 only;
6. evaluates once on the untouched test set;
7. saves the model, metrics, predictions, and confusion matrix.

## Setup

From the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r project_01_classical_baseline/requirements.txt
```

Then open the checkpoint notebook, select the `.venv` kernel, and choose **Run All**.

## Generated evidence

After a successful run:

```text
models/project_01_classical_baseline.joblib
results/project_01_checkpoint_metrics.json
results/project_01_test_predictions.csv
results/project_01_confusion_matrix.png
```

The included dataset is deliberately small and intended for learning the workflow. Its scores are not evidence of real-world performance.
