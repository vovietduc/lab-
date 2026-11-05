# Evaluation procedure

The official 10,000-image test split is used after training and checkpoint selection.

Reported metrics:
- Accuracy
- Macro-F1
- Cross-entropy test loss
- Per-class precision, recall and F1
- Confusion matrix
- Inference latency and parameter count

The evaluation script also saves random predictions and misclassified examples. Test metrics must not be used to tune hyperparameters.