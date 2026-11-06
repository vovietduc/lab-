# FashionMNIST Classification — MLP vs CNN

Bài thực hành Chapter 2 – Artificial Neural Networks (Practice 1), triển khai bằng Python và PyTorch. Phạm vi phiên bản này chỉ gồm hai mô hình MLP và CNN; ResNet18 đã được loại khỏi nội dung so sánh.

## Mục tiêu
- Thực hành Tensor, Dataset, DataLoader và transforms.
- Xây dựng MLP và CNN phân loại FashionMNIST 10 lớp.
- Hiểu training loop: forward → loss → backward → optimizer update.
- Theo dõi train/validation loss và accuracy; đánh giá test bằng Accuracy, Precision, Recall, Macro-F1 và Confusion Matrix.
- Lưu/nạp checkpoint, dự đoán ảnh và so sánh chất lượng với chi phí tính toán.

## Dữ liệu
FashionMNIST gồm 70.000 ảnh grayscale 28×28 thuộc 10 lớp: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

| Split | Samples | Purpose |
|---|---:|---|
| Train | 50,000 | Cập nhật trọng số |
| Validation | 10,000 | Theo dõi và chọn checkpoint |
| Test | 10,000 | Đánh giá cuối cùng |

60.000 ảnh train gốc được chia train/validation theo seed 42. Test set được giữ riêng. Ảnh được chuyển thành tensor và chuẩn hóa bằng mean/std tính từ train split. CNN nhận tensor (B,1,28,28); MLP Flatten ảnh thành vector 784 chiều.

## Hai mô hình
**MLP:** Flatten → Linear(784,256) → ReLU → Dropout → Linear(256,128) → ReLU → Dropout → Linear(128,10). MLP đơn giản, phù hợp làm baseline; Flatten làm mất cấu trúc không gian 2D.

**CNN:** ba tầng convolution với 32, 64, 128 kênh; Batch Normalization, ReLU, Max Pooling; Adaptive Average Pooling và classifier 1152 → 128 → 10. CNN khai thác đặc trưng cục bộ và quan hệ không gian pixel.

## Cấu trúc thư mục
```text
fashionmnist_lab/
├── configs/              # Cấu hình CNN và MLP
├── data/                 # FashionMNIST
├── docs/                 # Tài liệu dữ liệu, mô hình, train, evaluate, audit
├── notebooks/            # Notebook thử nghiệm
├── reports/
│   ├── cnn/              # Checkpoint, metrics, plots CNN
│   ├── mlp/              # Checkpoint, metrics, plots MLP
│   └── final_report/     # report.md và report.docx
├── src/                  # dataset, data_audit, model, train, evaluate, predict, utils
├── tests/
├── requirements.txt
└── README.md
```

## Cài đặt
Khuyến nghị Python 3.10 hoặc 3.11. Mở PowerShell tại thư mục fashionmnist_lab:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Cài PyTorch CPU/CUDA phù hợp theo https://pytorch.org/get-started/locally/

## Chạy chương trình
```powershell
python -m src.model
python -m src.dataset
python -m src.data_audit
python -m src.train --dry-run
python -m src.train
python -m src.train --config configs/mlp.yaml
python -m src.evaluate
python -m src.evaluate --config configs/mlp.yaml
pytest -q
```
Dry run chạy một epoch trên tập mẫu nhỏ và không ghi đè kết quả chính.

## Cấu hình thực nghiệm
| Parameter | Value |
|---|---:|
| Seed | 42 |
| Batch size | 128 |
| Maximum epochs | 15 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | CrossEntropyLoss |
| Weight decay | 0.0001 |
| Early stopping patience | 4 |
| MLP hidden sizes | 256, 128 |
| CNN channels | 32, 64, 128 |

## Kết quả thực nghiệm hiện có
| Metric | MLP | CNN |
|---|---:|---:|
| Best validation accuracy | 89.48% | 92.05% |
| Test accuracy | 88.65% | 91.30% |
| Test macro-F1 | 88.62% | 91.32% |
| Test loss | 0.3227 | 0.2473 |
| Training time | 21.88 s | 201.26 s |
| Inference latency | 0.0673 ms/image | 0.2684 ms/image |
| Parameters | 235,146 | 241,994 |
| Epochs run | 15 | 12 |

**Nhận xét:** CNN cao hơn MLP 2.65 điểm phần trăm về test accuracy và có macro-F1 cao hơn; test loss thấp hơn. MLP huấn luyện nhanh hơn và có latency thấp hơn trong lần đo hiện tại. Hai mô hình có số tham số gần nhau. Thời gian phụ thuộc phần cứng và điều kiện đo.

Shirt là lớp khó nhất (F1 MLP 0.6991; CNN 0.7506). CNN tốt hơn trên phần lớn lớp; riêng Bag, MLP đạt F1 0.9758, nhỉnh hơn CNN 0.9711.

## Data Audit
Chạy `python -m src.data_audit` để kiểm tra số lượng mẫu, kích thước ảnh, kiểu dữ liệu, miền pixel, nhãn, phân chia train/validation, phân phối lớp và thống kê chuẩn hóa chỉ từ train split. Script in kết quả PASS/FAIL; chỉ ghi nhận audit đạt sau khi chạy thực tế. Xem `docs/data_audit.md`.

## File đầu ra
Mỗi thư mục reports/mlp/ và reports/cnn/ có history.json, train_summary.json, test_metrics.json, classification_report.txt, training_curves.png, confusion_matrix.png, predictions_random.png, predictions_wrong.png và checkpoint .pth. Báo cáo tổng hợp nằm trong reports/final_report/.

## Ý nghĩa chỉ số
- Accuracy: tỷ lệ dự đoán đúng trên toàn bộ mẫu.
- Precision: tỷ lệ dự đoán đúng trong các mẫu được dự đoán thuộc lớp.
- Recall: tỷ lệ mẫu thuộc lớp được nhận diện đúng.
- F1-score: trung bình điều hòa precision và recall; Macro-F1 lấy trung bình đều trên các lớp.
- Confusion Matrix: số lượng dự đoán theo lớp thật và lớp dự đoán.
- Loss: CrossEntropyLoss đo sai khác giữa logits và nhãn.

## Tài liệu tham khảo
1. https://github.com/zalandoresearch/fashion-mnist
2. https://pytorch.org/tutorials/
3. https://pytorch.org/docs/stable/nn.html
4. https://pytorch.org/docs/stable/data.html