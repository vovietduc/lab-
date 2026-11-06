# Training procedure – Task 2: Loss visualization

For each mini-batch:

1. Move input images and labels to the selected device.
2. Clear previous gradients with `optimizer.zero_grad()`.
3. Compute logits in the forward pass.
4. Calculate `CrossEntropyLoss`.
5. Run `loss.backward()` to compute gradients.
6. Run `optimizer.step()` to update parameters.

Validation is performed after each epoch. The checkpoint with the best validation accuracy is retained. Early stopping uses patience 4. Default optimizer is Adam with learning rate 0.001, weight decay 0.0001 and batch size 128.

## Loss history and plots
The training loop records `train_loss`, `val_loss`, `train_acc` and `val_acc` after every epoch in `reports/<model>/history.json`. The `plot_history()` function writes `reports/<model>/training_curves.png`.

For Task 2, inspect the Train Loss and Validation Loss curves:
- Both decreasing indicates that training is progressing.
- Training loss decreasing while validation loss rises persistently can indicate overfitting.
- Interpret the trend across epochs rather than a single point.

The loss curves are a required deliverable, not merely an optional model-comparison chart.