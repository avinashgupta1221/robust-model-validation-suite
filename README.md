# Robust Model Validation Suite with Regularization Experiments

A complete, reproducible neural-network experiment comparing a baseline model with Dropout, L2 weight decay, and Early Stopping. It uses the built-in Breast Cancer Wisconsin dataset from scikit-learn, so there is no manual dataset download.

## Project structure

```text
model_validation_suite/
├── src/train.py                 # Training, evaluation, plots, and report generation
├── results/                     # Generated charts and CSV metrics
├── report/experiment_report.md  # Generated short experiment report
├── requirements.txt
└── README.md
```

## Run it (Windows)

1. Extract the ZIP and open the project folder in VS Code.
2. Open Terminal (**Terminal → New Terminal**).
3. Install packages:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run:
   ```powershell
   python src/train.py
   ```

## Outputs
- `results/model_comparison.csv`: test accuracy, precision, recall, F1, ROC-AUC, best validation loss, and epochs trained.
- `results/learning_curves.png`: train/validation loss curves for all experiments.
- `results/metric_comparison.png`: bar chart comparing test metrics.
- `report/experiment_report.md`: automatically generated brief findings report.

## Experiments
- Baseline neural network
- Dropout (0.30)
- L2 weight decay (0.001)
- Early stopping (patience 15; restores best validation-loss weights)

The split is stratified: 70% train, 15% validation, and 15% test. The scaler is fitted on training data only. A fixed random seed (42) is used for reproducibility.
