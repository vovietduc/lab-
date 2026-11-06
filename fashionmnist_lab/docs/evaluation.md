# Evaluation procedure – Task 3: Predicted vs Actual

The official 10,000-image test split is used after training and checkpoint selection. Test data must not be used to tune hyperparameters.

Reported metrics:
- Accuracy and Macro-F1
- Cross-entropy test loss
- Per-class precision, recall and F1
- Confusion matrix
- Inference latency and parameter count

## Predicted-versus-actual images
The evaluation script also saves:
- `predictions_random.png`: randomly selected test images.
- `predictions_wrong.png`: misclassified test images, when any exist.

Each image title uses the format **Predicted (Actual)**. Green indicates a correct prediction; red indicates a mismatch. This is the visual evidence for Task 3.

Files are written to `reports/mlp/` and `reports/cnn/`. The report and notebook should embed these images so that the task is visibly demonstrated.