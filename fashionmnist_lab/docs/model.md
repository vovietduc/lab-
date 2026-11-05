# Model architectures

## MLP

`Flatten → Linear(784,256) → ReLU → Dropout → Linear(256,128) → ReLU → Dropout → Linear(128,10)`.

The MLP is a fully connected baseline. Flattening simplifies the input but discards the explicit 2D neighborhood structure.

## CNN

Three convolution blocks use 32, 64 and 128 channels, with Batch Normalization, ReLU and Max Pooling. Adaptive Average Pooling produces a 3×3 feature map; the classifier is `1152 → 128 → 10`.

The CNN preserves spatial structure and learns local visual patterns through convolution filters.

Both models output 10 logits and are trained from scratch.