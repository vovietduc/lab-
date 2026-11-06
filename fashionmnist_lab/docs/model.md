# Model building – Task 1: Architecture experiment

Practice 1 requires an experiment with a network structure **or** hyperparameters. This project chooses a controlled architecture experiment using two networks:

## MLP
`Flatten → Linear(784,256) → ReLU → Dropout(0.2) → Linear(256,128) → ReLU → Dropout(0.2) → Linear(128,10)`.

The MLP is a fully connected network. Flattening converts each 28×28 image into 784 values and removes the explicit 2D neighborhood layout.

## CNN
Three convolution blocks use 32, 64 and 128 channels, with Batch Normalization, ReLU and Max Pooling. Adaptive Average Pooling produces a 3×3 feature map; the classifier is `1152 → 128 → 10`.

The CNN preserves spatial structure and learns local visual patterns through convolution filters.

## Controlled experiment
Both configurations use the same FashionMNIST split, seed, batch size (128), Adam optimizer, learning rate (0.001), weight decay (0.0001) and maximum epochs (15). The architecture is the main experimental factor. Configuration files: `configs/mlp.yaml` and `configs/cnn.yaml`.

The comparison is included only to demonstrate Task 1; the lab's other required tasks are loss visualization and predicted-versus-actual image display.