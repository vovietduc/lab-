# Data pipeline – FashionMNIST

- Dataset: FashionMNIST, 70,000 grayscale images, 28×28 pixels, 10 classes.
- Original training split: 60,000 images; split into 50,000 train and 10,000 validation using seed 42.
- Official test split: 10,000 images; reserved for final evaluation.
- Images are converted to tensors and normalized with mean/std computed from the training split only.
- CNN input shape: `(B, 1, 28, 28)`. MLP flattens each image to 784 features.
- DataLoaders use batch size 128 by default.

This split prevents test data from influencing training or model selection.