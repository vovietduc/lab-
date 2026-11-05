# Project Description – FashionMNIST Classification

## 1. Mục tiêu
Bài thực hành thuộc Chapter 2 – Artificial Neural Networks. Xây dựng pipeline phân loại ảnh thời trang bằng PyTorch và so sánh hai mô hình huấn luyện từ đầu: MLP (Multi-Layer Perceptron) và CNN (Convolutional Neural Network). Không sử dụng ResNet hoặc trọng số pretrained.

## 2. Bài toán và dữ liệu
Đầu vào là ảnh grayscale 28×28 thuộc FashionMNIST. Đầu ra là một trong 10 lớp: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. Dataset có 60.000 ảnh train gốc và 10.000 ảnh test. Chia train gốc thành 50.000 train và 10.000 validation với seed 42. Test chỉ dùng đánh giá cuối cùng.

Tensor đầu vào có shape (B,1,28,28), float32, được chuẩn hóa bằng mean/std của tập train. Đầu ra là 10 logits.

## 3. Phạm vi mô hình
### MLP
Flatten ảnh thành vector 784 chiều; hai hidden layers mặc định có 256 và 128 neuron; ReLU, Dropout và output 10 lớp.

### CNN
Ba tầng convolution với số kênh 32, 64, 128; Batch Normalization, ReLU và Max Pooling; Adaptive Average Pooling và classifier 10 lớp.

## 4. Yêu cầu chức năng
1. Tải FashionMNIST và tạo DataLoader.
2. Xây dựng MLP và CNN có cùng định dạng đầu vào/đầu ra.
3. Huấn luyện theo mini-batch với forward, loss, backward và optimizer.
4. Theo dõi train/validation loss và accuracy.
5. Lưu checkpoint tốt nhất theo validation accuracy.
6. Đánh giá test accuracy, macro-F1 và loss.
7. Vẽ training curves, confusion matrix và ảnh dự đoán.
8. Nạp checkpoint để dự đoán ảnh đơn.
9. Cung cấp README và báo cáo kết quả.

## 5. Yêu cầu phi chức năng
- Code chia module, dễ đọc và có thể chạy lại.
- Seed cố định để tăng khả năng tái lập.
- Cấu hình mô hình và hyperparameter trong YAML.
- Chạy CPU; hỗ trợ CUDA nếu cài PyTorch tương ứng.
- Không ghi số liệu trước khi thực sự chạy thí nghiệm.

## 6. Tiêu chí so sánh
Accuracy, Macro-F1, Test Loss, thời gian huấn luyện, latency suy luận, số lượng tham số và khả năng nhận diện từng lớp qua confusion matrix.
