# Training procedure

For each mini-batch:

1. Move input images and labels to the selected device.
2. Clear previous gradients with `optimizer.zero_grad()`.
3. Compute logits in the forward pass.
4. Calculate `CrossEntropyLoss`.
5. Run `loss.backward()` to compute gradients.
6. Run `optimizer.step()` to update parameters.

Validation is performed after each epoch. The checkpoint with the best validation accuracy is retained. Early stopping uses patience 4. Default optimizer is Adam with learning rate 0.001, weight decay 0.0001 and batch size 128.