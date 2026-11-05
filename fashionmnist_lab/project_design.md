# Project Design – MLP vs CNN trên FashionMNIST

## 1. Kiến trúc tổng thể
FashionMNIST → Dataset/Split/Normalize → DataLoader → (MLP hoặc CNN) → CrossEntropyLoss → Backpropagation/Adam → Validation/Early Stopping → Checkpoint → Test Evaluation → Metrics và biểu đồ.

## 2. Thiết kế dữ liệu
- Dùng torchvision FashionMNIST.
- Chia train/validation từ 60.000 ảnh train gốc bằng seed 42.
- Giữ 10.000 ảnh test chuẩn độc lập.
- Tính mean/std trên tập train và dùng cùng thống kê cho validation/test.
- Train DataLoader shuffle=True; validation/test shuffle=False.

## 3. Thiết kế MLP
Input (B,1,28,28) → Flatten → 784 features → Linear(256) → ReLU → Dropout → Linear(128) → ReLU → Dropout → Linear(10). MLP không có convolution nên không giữ quan hệ pixel lân cận sau Flatten.

## 4. Thiết kế CNN
Input (B,1,28,28) → Conv(1,32) → BatchNorm → ReLU → MaxPool → Conv(32,64) → BatchNorm → ReLU → MaxPool → Conv(64,128) → BatchNorm → ReLU → AdaptiveAvgPool(3,3) → Flatten → Linear(128) → ReLU → Dropout → Linear(10). CNN giữ cấu trúc không gian và học bộ lọc đặc trưng từ dữ liệu.

## 5. Thiết lập huấn luyện
| Thành phần | Thiết lập |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epoch tối đa | 15 |
| Early stopping | patience = 4 |
| Seed | 42 |
| Checkpoint | validation accuracy tốt nhất |

## 6. Module
- src/dataset.py: tải dữ liệu, chia tập, chuẩn hóa, DataLoader.
- src/model.py: MLPClassifier, CNNClassifier, build_model.
- src/train.py: training loop, validation, early stopping, checkpoint.
- src/evaluate.py: test metrics, confusion matrix, prediction examples.
- src/predict.py: nạp checkpoint và dự đoán ảnh mới.
- src/utils.py: seed, đo latency, lưu JSON, vẽ hình.
- configs/: cấu hình CNN mặc định và MLP.
- reports/: checkpoint, metrics, biểu đồ và báo cáo.

## 7. Quy trình
1. Smoke test: python -m src.model và python -m src.dataset.
2. Dry run: python -m src.train --dry-run.
3. Train CNN: python -m src.train.
4. Train MLP: python -m src.train --config configs/mlp.yaml.
5. Evaluate từng mô hình.
6. Chạy pytest.
7. Tổng hợp số liệu và hình ảnh vào reports/final_report.

## 8. Đánh giá
Accuracy, Macro-F1, Cross-Entropy Loss, Confusion Matrix, training time, inference latency và parameter count. Không dùng số liệu minh họa làm kết quả; chỉ ghi kết quả sau khi train/evaluate thành công.
