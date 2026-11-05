# project_design.md – Thiết kế hệ thống Lab 1

## 1. Cấu trúc thư mục (Modular)
```
fashionmnist_lab/
├── project_description.md      # Định nghĩa bài toán, dữ liệu, metric
├── project_design.md           # Tài liệu này
├── README.md                   # Báo cáo, hướng dẫn chạy, kết quả
├── AI_usage.md                 # Nhật ký sử dụng và kiểm duyệt AI
├── requirements.txt, pytest.ini, .gitignore
├── configs/
│   ├── default.yaml            # ResNet-18 + MLP head
│   └── mlp.yaml                # Baseline MLP
├── src/
│   ├── dataset.py              # Data Pipeline
│   ├── model.py                # Backbone -> Head, MLP baseline
│   ├── train.py                # Vòng lặp huấn luyện
│   ├── evaluate.py             # Đánh giá trên test
│   ├── predict.py              # Suy luận một ảnh -> JSON
│   └── utils.py                # Seed, đo lường, vẽ biểu đồ
├── tests/                      # pytest: dataset, model, train smoke
├── notebooks/                  # Phân tích dữ liệu và thử nghiệm (EDA)
├── reports/                    # Biểu đồ, metrics, ảnh dự đoán (sinh ra khi chạy)
└── fashion_mnist_model.pth     # Trọng số mô hình (sinh ra khi chạy train)
```

## 2. Luồng dữ liệu
`FashionMNIST (torchvision) -> split train/val -> chuẩn hóa z-score (mean/std của train) -> TensorDataset -> DataLoader -> Model -> Loss -> Optimizer`

## 3. Kiến trúc mô hình: Backbone -> Feature Vector -> Head
- **Backbone:** ResNet-18 pretrained (ImageNet), bỏ lớp FC cuối, đầu ra vector 512 chiều.
- **Task Head (MLP):** `Flatten -> Linear(512,256) -> ReLU -> Dropout(0.3) -> Linear(256,10)`.
- **Tiền xử lý trong model:** phóng ảnh 28×28 lên 64×64 (bilinear), nhân 1 kênh thành 3 kênh.
- **Chiến lược:** Feature Extraction (đóng băng backbone, chỉ train head); có thể đổi `freeze_backbone: false` để Fine-Tuning với learning rate nhỏ.
- **Quyết định thiết kế:** khi đóng băng backbone, giữ BatchNorm ở chế độ eval để thống kê ImageNet không bị cập nhật.

## 4. Huấn luyện
| Thành phần | Lựa chọn |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr = 1e-3), hỗ trợ AdamW, SGD |
| Điều hòa | Dropout 0.3, Early Stopping (patience = 4), weight decay tùy chọn |
| Hiệu năng | Mixed Precision (AMP) khi có GPU CUDA |
| Checkpoint | Giữ trọng số có **val accuracy cao nhất**, lưu cùng cấu hình, mean, std, tên lớp vào `fashion_mnist_model.pth` |

## 5. Bảng siêu tham số
| Nhóm | Tham số | Nơi cấu hình |
|---|---|---|
| Tối ưu hóa và huấn luyện | learning rate, batch size, epochs | `train.lr`, `data.batch_size`, `train.epochs` |
| Kiến trúc | hidden size, resize_to, freeze_backbone | `model.*` |
| Thuật toán tối ưu | optimizer, weight decay | `train.optimizer`, `train.weight_decay` |
| Điều hòa | dropout, patience | `model.dropout`, `train.patience` |

## 6. Quy trình triển khai từng module (8 bước)
Describe → Generate → Review Code → Run (Smoke/Dry) → Normal Run → Test → Refactor → Commit

| Bước | Lệnh |
|---|---|
| Smoke Run | `python -m src.dataset` và `python -m src.model` |
| Dry Run | `python -m src.train --dry-run` |
| Normal Run | `python -m src.train` |
| Test | `pytest` và `python -m src.evaluate` |

## 7. Hạn chế đã biết của thiết kế
- Backbone pretrained trên ảnh RGB tự nhiên, trong khi dữ liệu là ảnh xám nhỏ phóng lên, nên đặc trưng đóng băng có thể không tối ưu.
- Chuẩn hóa theo thống kê của FashionMNIST, không theo thống kê ImageNet mà backbone được huấn luyện.
