# Training workflow

```text
FashionMNIST
    ↓
Transforms + normalization
    ↓
Dataset / DataLoader
    ↓
MLP or CNN
    ↓
Forward → CrossEntropyLoss → Backward → Adam update
    ↓
Validation + checkpoint selection
    ↓
Test evaluation + metrics + plots
```

Both architectures use the same data split and evaluation pipeline.