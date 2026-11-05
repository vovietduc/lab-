# Run the experiment

Run commands from the `fashionmnist_lab` directory:

```powershell
python -m src.model
python -m src.dataset
python -m src.train --dry-run
python -m src.train
python -m src.train --config configs/mlp.yaml
python -m src.evaluate
python -m src.evaluate --config configs/mlp.yaml
pytest -q
```

The default configuration trains CNN. The `mlp.yaml` configuration trains MLP. Results are written to `reports/cnn/` and `reports/mlp/` respectively.