# FashionMNIST Classification – So sánh MLP và CNN

Dự án thực hành Chapter 2 – Artificial Neural Networks. Mục tiêu là xây dựng và so sánh hai kiến trúc neural network bằng PyTorch trên FashionMNIST: Multi-Layer Perceptron (MLP) và Convolutional Neural Network (CNN). Cả hai được huấn luyện từ đầu; dự án không sử dụng ResNet hay trọng số pretrained.

## 1. Mục tiêu học tập
- Làm quen với Tensor, Dataset, DataLoader và tiền xử lý dữ liệu trong PyTorch.
- Hiểu forward pass, CrossEntropyLoss, backpropagation và optimizer.
- Xây dựng hai mô hình MLP và CNN cho phân loại ảnh 10 lớp.
- Theo dõi train/validation loss và accuracy.
- Lưu checkpoint, nạp lại mô hình và dự đoán ảnh.
- So sánh hiệu quả và chi phí tính toán của hai kiến trúc.

## 2. Dữ liệu FashionMNIST
FashionMNIST có 70.000 ảnh grayscale kích thước 28×28 thuộc 10 lớp: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag và Ankle boot. Dữ liệu gồm 60.000 ảnh train gốc và 10.000 ảnh test. Tập train gốc được chia thành 50.000 ảnh train và 10.000 ảnh validation với seed 42. Tập test được giữ riêng để đánh giá cuối cùng.

Input tensor có shape (B, 1, 28, 28), được chuẩn hóa bằng mean/std tính trên tập train. Output của mỗi mô hình là 10 logits.

## 3. Mô hình

### MLP
Ảnh 28×28 được Flatten thành vector 784 chiều. Kiến trúc mặc định: Linear(784,256) → ReLU → Dropout → Linear(256,128) → ReLU → Dropout → Linear(128,10). MLP đơn giản, dễ huấn luyện nhưng không giữ cấu trúc không gian sau khi Flatten.

### CNN
CNN giữ ảnh ở dạng 2D và học đặc trưng không gian bằng convolution. Kiến trúc gồm ba tầng convolution với 32, 64 và 128 kênh; Batch Normalization, ReLU và Max Pooling; Adaptive Average Pooling; cuối cùng là Fully Connected classifier 10 lớp.

## 4. Quy trình huấn luyện
1. Tải FashionMNIST bằng torchvision.
2. Chia train/validation, chuẩn hóa ảnh và tạo DataLoader.
3. Forward pass để tính logits.
4. Tính CrossEntropyLoss.
5. Gọi backward để tính gradient.
6. Adam cập nhật trọng số.
7. Đánh giá validation mỗi epoch và lưu checkpoint có validation accuracy tốt nhất.
8. Đánh giá test và sinh biểu đồ, confusion matrix, ảnh dự đoán.

Cấu hình mặc định: seed 42, batch size 128, tối đa 15 epochs, Adam, learning rate 0.001 và Early Stopping patience 4. Cấu hình nằm trong configs/default.yaml (CNN) và configs/mlp.yaml (MLP).

## 5. Cấu trúc thư mục
- configs/: cấu hình CNN và MLP.
- src/dataset.py: tải dữ liệu, chia tập, chuẩn hóa và DataLoader.
- src/model.py: định nghĩa MLPClassifier, CNNClassifier và build_model.
- src/train.py: training loop, validation, early stopping và checkpoint.
- src/evaluate.py: test metrics và trực quan hóa.
- src/predict.py: dự đoán ảnh đơn.
- src/utils.py: seed, đếm tham số, đo latency, lưu JSON và vẽ biểu đồ.
- tests/: kiểm thử model, dataset và training smoke.
- reports/cnn/: checkpoint và kết quả CNN.
- reports/mlp/: checkpoint và kết quả MLP.
- reports/final_report/: báo cáo Markdown và DOCX.

## 6. Cài đặt
Dùng Python 3.10 hoặc 3.11. Mở terminal ở thư mục fashionmnist_lab và chạy:

Windows PowerShell:
1. python -m venv .venv
2. .venv\Scripts\Activate.ps1
3. python -m pip install --upgrade pip
4. pip install -r requirements.txt

Cài phiên bản PyTorch CPU/CUDA phù hợp theo https://pytorch.org/get-started/locally/

## 7. Chạy chương trình

Kiểm tra nhanh:
- python -m src.model
- python -m src.dataset

Dry run (một epoch trên dữ liệu nhỏ, không ghi đè kết quả thật):
- python -m src.train --dry-run

Huấn luyện CNN:
- python -m src.train

Huấn luyện MLP:
- python -m src.train --config configs/mlp.yaml

Đánh giá CNN và MLP:
- python -m src.evaluate
- python -m src.evaluate --config configs/mlp.yaml

Kiểm thử:
- pytest -q

## 8. Kết quả được tạo
Mỗi thư mục reports/cnn hoặc reports/mlp có thể chứa:
- training_curves.png: train/validation loss và accuracy theo epoch.
- history.json: lịch sử từng epoch.
- train_summary.json: thời gian train, epoch tốt nhất và peak VRAM khi train.
- test_metrics.json: test accuracy, macro-F1, test loss, latency và số tham số.
- classification_report.txt: precision, recall, F1-score theo từng lớp.
- confusion_matrix.png: ma trận nhầm lẫn.
- predictions_random.png: ảnh test và nhãn dự đoán.
- predictions_wrong.png: các ảnh dự đoán sai.
- checkpoint .pth: trọng số mô hình cùng cấu hình và thống kê chuẩn hóa.

## 9. Bảng so sánh thực nghiệm
Chạy đầy đủ hai mô hình trước khi điền số liệu. Không sử dụng số liệu giả định.

| Chỉ số | MLP | CNN |
|---|---:|---:|
| Validation Accuracy | Chưa chạy | Chưa chạy |
| Test Accuracy | Chưa chạy | Chưa chạy |
| Macro-F1 | Chưa chạy | Chưa chạy |
| Test Loss | Chưa chạy | Chưa chạy |
| Training time (s) | Chưa chạy | Chưa chạy |
| Inference latency (ms/ảnh) | Chưa chạy | Chưa chạy |
| Số tham số | Chạy script | Chạy script |

Nhận xét cần dựa trên kết quả thực nghiệm: mô hình nào có accuracy và macro-F1 cao hơn; mô hình nào nhanh hơn; train/validation curves có dấu hiệu overfitting hay không; lớp nào dễ bị nhầm qua confusion matrix.

## 10. Tài liệu tham khảo
1. Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
2. PyTorch Tutorials: https://pytorch.org/tutorials/
3. torch.nn: https://pytorch.org/docs/stable/nn.html
4. Data Loading: https://pytorch.org/docs/stable/data.html
