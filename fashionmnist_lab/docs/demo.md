# Run the Practice 1 experiment

Run commands from the `fashionmnist_lab` directory:

```powershell
python -m src.model
python -m src.dataset
python -m src.data_audit
python -m src.train --dry-run

# Task 1: architecture experiment
python -m src.train --config configs/mlp.yaml
python -m src.evaluate --config configs/mlp.yaml
python -m src.train --config configs/cnn.yaml
python -m src.evaluate --config configs/cnn.yaml

python -m pytest -q
```

## Check the three required tasks

1. **Architecture experiment:** inspect `configs/mlp.yaml` and `configs/cnn.yaml`; both use the same data split and core training settings while changing the network structure.
2. **Visualize loss:** open `reports/mlp/training_curves.png` and `reports/cnn/training_curves.png`. Each chart includes train and validation loss by epoch.
3. **Predicted vs actual:** open `reports/mlp/predictions_random.png`, `reports/cnn/predictions_random.png` and the corresponding `predictions_wrong.png` files. Titles show predicted class and actual class.

The default configuration trains CNN. Results are written to `reports/cnn/`; MLP results are written to `reports/mlp/`. Re-run the notebook's result and visualization cells after training/evaluation so that notebook output and report figures match the latest files.