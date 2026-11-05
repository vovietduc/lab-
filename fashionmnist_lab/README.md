# FashionMNIST Classification (Lab 1)

## 1. Project Overview
Dự án huấn luyện mô hình Deep Learning phân loại ảnh quần áo trong bộ dữ liệu FashionMNIST bằng PyTorch. Mô hình chính gồm **Backbone ResNet-18 (pretrained) + MLP Task Head**, được so sánh với baseline MLP thuần. Mục tiêu: xây dựng pipeline đầy đủ (dữ liệu → mô hình → huấn luyện → đánh giá → lưu/nạp model) theo cấu trúc modular.

## 2. Problem Statement
- **Bài toán:** phân loại ảnh xám 28×28 vào 10 lớp (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **Đầu vào:** tensor `(B, 1, 28, 28)`. **Đầu ra:** logits `(B, 10)`.
- **Thách thức:** các lớp áo rất giống nhau; ảnh nhỏ và ít chi tiết. Chi tiết xem `project_description.md`.

## 3. Main Features
- Kiến trúc modular (`dataset.py`, `model.py`, `train.py`, `evaluate.py`, `predict.py`).
- Cấu hình bằng YAML, đổi mô hình hoặc siêu tham số không cần sửa code.
- Mixed Precision (AMP) khi có GPU, Early Stopping, lưu checkpoint tốt nhất theo validation.
- Logging và báo cáo tự động: đường cong loss/accuracy, confusion matrix, ảnh dự đoán, metrics JSON.
- Smoke Run, Dry Run và unit test (pytest).

## 4. System Architecture
```
FashionMNIST ──► Dataset/DataLoader ──► Model (Backbone → Head) ──► Trainer ──► Checkpoint (.pth)
(torchvision)    (split, normalize)     ResNet-18 + MLP             Loss, Adam,         │
                                                                    Early Stopping      ▼
                                                           Evaluate ──► reports/ (biểu đồ, metrics)
```
Chi tiết kiến trúc và quyết định thiết kế: `project_design.md`.

## 5. Folder Structure
```
fashionmnist_lab/
├── configs/  (default.yaml, mlp.yaml)
├── src/      (dataset.py, model.py, train.py, evaluate.py, predict.py, utils.py)
├── tests/    (test_dataset.py, test_model.py, test_train_smoke.py)
├── notebooks/ (phân tích dữ liệu và thử nghiệm)
├── reports/  (kết quả sinh ra khi chạy)
├── project_description.md, project_design.md, README.md, AI_usage.md
└── fashion_mnist_model.pth   (sinh ra sau khi train)
```

## 6. Installation / Run / Test
```bash
# Môi trường (khuyến nghị cài PyTorch bản CUDA theo pytorch.org/get-started/locally)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Smoke Run: truyền tensor giả / vài ảnh để bắt lỗi shape
python -m src.model
python -m src.dataset

# 2) Dry Run: 1 epoch trên 50 ảnh (không ghi đè kết quả thật)
python -m src.train --dry-run

# 3) Normal Run: huấn luyện đầy đủ, tạo fashion_mnist_model.pth và reports/resnet18/
python -m src.train
python -m src.train --config configs/mlp.yaml          # baseline để so sánh

# 4) Đánh giá trên tập test
python -m src.evaluate
python -m src.evaluate --config configs/mlp.yaml

# 5) Unit test
pytest
```

## 7. Example Input - Output
**Input:** tensor `(1, 1, 28, 28)` đã chuẩn hóa z-score (từ ảnh xám 28×28).

```bash
python -m src.predict --image sample.png
```
**Output** (JSON; giá trị dưới đây chỉ minh họa định dạng, kết quả thật phụ thuộc mô hình đã train):
```json
{
  "label": 9,
  "class": "Ankle boot",
  "confidence": 0.97,
  "probabilities": {"T-shirt/top": 0.0, "Trouser": 0.0, "...": 0.0, "Ankle boot": 0.97}
}
```

## 8. Results
> Điền bảng sau khi chạy `src.train` và `src.evaluate` cho từng cấu hình. Số liệu lấy từ `reports/<model>/test_metrics.json` và `train_summary.json`.

| Mô hình | Val Acc | Test Acc | Macro-F1 | Test Loss | Latency (ms/ảnh) | Peak VRAM (MB) | Tham số huấn luyện |
|---|---|---|---|---|---|---|---|
| MLP baseline | | | | | | | |
| ResNet-18 + MLP head (đóng băng) | | | | | | | |
| ResNet-18 fine-tune (tùy chọn) | | | | | | | |

**Hình ảnh kết quả** (sinh trong `reports/<model>/`): `training_curves.png` (loss/accuracy), `confusion_matrix.png`, `predictions_random.png`, `predictions_wrong.png`.

**Nhận xét** (viết sau khi có số liệu): so sánh các mô hình; cặp lớp nào bị nhầm nhiều nhất; dấu hiệu overfitting/underfitting từ đường cong loss.

## 9. Limitations
- Backbone pretrained trên ảnh RGB tự nhiên, còn dữ liệu là ảnh xám nhỏ phóng lên 64×64; đặc trưng đóng băng có thể chưa tối ưu.
- Chuẩn hóa theo thống kê FashionMNIST, không theo thống kê ImageNet.
- Chưa dùng data augmentation và learning rate scheduler.
- Các lớp áo giống nhau vẫn là nguồn nhầm lẫn chính.

## 10. Future Improvements
- Fine-tuning một phần backbone với learning rate nhỏ (`freeze_backbone: false`).
- Thêm data augmentation (lật ngang, xoay nhẹ) và LR scheduler.
- Thử CNN nhỏ huấn luyện từ đầu, hoặc backbone khác (MobileNet, EfficientNet).
- Theo dõi huấn luyện bằng TensorBoard.
