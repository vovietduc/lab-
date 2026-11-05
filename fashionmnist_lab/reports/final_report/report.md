# BÁO CÁO THỰC HÀNH – CHAPTER 2
## FASHIONMNIST IMAGE CLASSIFICATION: MLP VS CNN

**Học phần:** Artificial Neural Networks / Deep Learning  
**Công cụ:** Python, PyTorch, Torchvision, Matplotlib, Scikit-learn  
**Bộ dữ liệu:** FashionMNIST

---

## TÓM TẮT
Bài thực hành xây dựng pipeline phân loại ảnh thời trang FashionMNIST bằng PyTorch và so sánh hai kiến trúc mạng neural network: Multi-Layer Perceptron (MLP) và Convolutional Neural Network (CNN). Hai mô hình được huấn luyện từ đầu trên cùng bộ dữ liệu, cùng cách chia train/validation/test và cùng quy trình đánh giá. Báo cáo tập trung giải thích cách chuẩn bị dữ liệu, kiến trúc, forward pass, tính loss, backpropagation, tối ưu hóa và đánh giá mô hình.

**Lưu ý:** Các ô kết quả được để trống có chủ đích. Cần chạy thực nghiệm trên máy trước khi điền số liệu, chèn hình và đưa ra kết luận; không sử dụng kết quả giả định.

## 1. GIỚI THIỆU
Phân loại ảnh là bài toán xác định lớp đối tượng tương ứng với một ảnh đầu vào. FashionMNIST là bộ dữ liệu phù hợp để thực hành vì có cấu trúc đơn giản, gồm ảnh grayscale kích thước 28×28 và 10 nhãn thời trang. Mặc dù ảnh có độ phân giải thấp, một số lớp có hình dạng tương tự nhau nên mô hình vẫn cần học được đặc trưng hữu ích.

Mục tiêu của bài là cài đặt hai cách tiếp cận:
- MLP: biến ảnh thành vector một chiều và sử dụng các lớp fully connected.
- CNN: sử dụng convolution và pooling để học đặc trưng không gian của ảnh.

Qua đó, bài thực hành làm rõ sự khác biệt giữa mạng fully connected và mạng tích chập khi xử lý dữ liệu hình ảnh.

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. Multi-Layer Perceptron
MLP gồm input layer, một hoặc nhiều hidden layers và output layer. Mỗi neuron ở một lớp được kết nối với các neuron ở lớp kế tiếp. Với FashionMNIST, ảnh 28×28 được Flatten thành vector 784 phần tử. Mỗi lớp Linear thực hiện phép biến đổi tuyến tính, sau đó hàm kích hoạt ReLU tạo tính phi tuyến.

Ưu điểm của MLP là cấu trúc đơn giản, dễ triển khai và phù hợp làm mô hình baseline. Hạn chế là bước Flatten làm mất cấu trúc không gian hai chiều; mô hình không có cơ chế tích chập để khai thác trực tiếp các pixel lân cận.

### 2.2. Convolutional Neural Network
CNN được thiết kế cho dữ liệu có cấu trúc lưới như ảnh. Lớp convolution sử dụng các kernel để trích xuất đặc trưng cục bộ; các lớp đầu thường học cạnh và đường viền, các lớp sâu hơn có thể kết hợp thành đặc trưng phức tạp hơn. ReLU bổ sung tính phi tuyến. Max Pooling giảm kích thước feature map và giúp giảm chi phí tính toán.

CNN giữ được quan hệ không gian giữa các pixel và chia sẻ trọng số trong kernel. Vì vậy, CNN thường phù hợp với bài toán phân loại ảnh hơn MLP thuần. Tuy nhiên, hiệu quả cần được xác nhận bằng kết quả thực nghiệm.

### 2.3. Quy trình học trong PyTorch
Với mỗi mini-batch, mô hình thực hiện:
1. Forward: tính logits từ ảnh đầu vào.
2. Loss: CrossEntropyLoss đo sai lệch giữa logits và nhãn thật.
3. Backward: loss.backward() tính gradient theo các tham số.
4. Update: optimizer.step() cập nhật trọng số theo gradient.
5. Zero gradient: optimizer.zero_grad() xóa gradient cũ trước batch kế tiếp.

Quá trình lặp lại qua nhiều epoch. Validation được dùng để theo dõi khả năng tổng quát hóa và lựa chọn checkpoint; test set chỉ dùng để đánh giá cuối cùng.

## 3. DỮ LIỆU VÀ TIỀN XỬ LÝ

### 3.1. Mô tả FashionMNIST
Bộ dữ liệu có 70.000 ảnh grayscale, mỗi ảnh kích thước 28×28 pixel và thuộc một trong 10 lớp:

| Nhãn | Lớp | Nhãn | Lớp |
|---:|---|---:|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

### 3.2. Chia tập dữ liệu

| Tập | Số ảnh | Mục đích |
|---|---:|---|
| Train | 50.000 | Cập nhật trọng số |
| Validation | 10.000 | Theo dõi và chọn checkpoint |
| Test | 10.000 | Đánh giá cuối cùng |

Train và validation được tách từ 60.000 ảnh train gốc với seed 42. Test là tập test chính thức và không được dùng để điều chỉnh mô hình.

### 3.3. Chuẩn hóa
Ảnh được chuyển về float32 trong khoảng [0,1], sau đó chuẩn hóa theo công thức:

x_normalized = (x - mean_train) / std_train

Mean và standard deviation chỉ được tính từ phần train nhằm tránh rò rỉ thông tin từ validation/test. Tensor đầu vào có kích thước (B, 1, 28, 28).

## 4. THIẾT KẾ MÔ HÌNH

### 4.1. MLP
Kiến trúc sử dụng:
- Input: ảnh 1×28×28.
- Flatten: 784 đặc trưng.
- Fully connected: 784 → 256, ReLU, Dropout.
- Fully connected: 256 → 128, ReLU, Dropout.
- Output: 128 → 10 logits.

MLP xử lý toàn bộ vector pixel cùng lúc. Đây là kiến trúc baseline để quan sát hiệu quả của fully connected layers khi không có phép tích chập.

### 4.2. CNN
Kiến trúc sử dụng:
- Conv2D: 1 → 32 channels, kernel 3×3; BatchNorm, ReLU, MaxPool.
- Conv2D: 32 → 64 channels, kernel 3×3; BatchNorm, ReLU, MaxPool.
- Conv2D: 64 → 128 channels, kernel 3×3; BatchNorm, ReLU.
- Adaptive Average Pooling: đưa feature map về 3×3.
- Flatten và Fully Connected: 1152 → 128 → 10.

CNN nhận trực tiếp tensor ảnh 2D, học các bộ lọc từ dữ liệu và tổng hợp đặc trưng qua các tầng.

### 4.3. So sánh thiết kế

| Đặc điểm | MLP | CNN |
|---|---|---|
| Input | Flatten 784 chiều | Tensor ảnh 1×28×28 |
| Thành phần chính | Linear, ReLU | Conv, Pooling, ReLU |
| Không gian ảnh | Không giữ sau Flatten | Giữ cấu trúc không gian |
| Chia sẻ trọng số | Không theo kernel | Có trong convolution |
| Mục đích | Baseline đơn giản | Khai thác đặc trưng hình ảnh |

## 5. THIẾT LẬP THỰC NGHIỆM

Hai mô hình dùng cùng seed, cách chia dữ liệu và tập test.

| Tham số | Giá trị mặc định |
|---|---|
| Batch size | 128 |
| Epoch tối đa | 15 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | CrossEntropyLoss |
| Seed | 42 |
| Early stopping patience | 4 |
| MLP hidden sizes | 256, 128 |
| CNN channels | 32, 64, 128 |

Các tham số được cấu hình trong configs/default.yaml cho CNN và configs/mlp.yaml cho MLP.

## 6. KẾT QUẢ THỰC NGHIỆM

### 6.1. Bảng kết quả tổng hợp

| Chỉ số | MLP | CNN |
|---|---:|---:|
| Best validation accuracy | Chưa chạy | Chưa chạy |
| Test accuracy | Chưa chạy | Chưa chạy |
| Macro-F1 | Chưa chạy | Chưa chạy |
| Test loss | Chưa chạy | Chưa chạy |
| Training time (s) | Chưa chạy | Chưa chạy |
| Inference latency (ms/ảnh) | Chưa chạy | Chưa chạy |
| Số tham số | Chạy script | Chạy script |

Điền số liệu sau khi chạy hai lệnh train và evaluate. Nguồn số liệu là reports/mlp và reports/cnn.

### 6.2. Đường cong Loss và Accuracy
Chèn hai hình training_curves.png được sinh ra trong mỗi thư mục kết quả.

**Hình 1.** Đường cong loss và accuracy của MLP.  
[Chèn reports/mlp/training_curves.png]

**Hình 2.** Đường cong loss và accuracy của CNN.  
[Chèn reports/cnn/training_curves.png]

Nhận xét:
- So sánh train loss và validation loss.
- Kiểm tra khoảng cách train accuracy và validation accuracy.
- Nếu train loss giảm nhưng validation loss tăng, có thể mô hình đang overfitting.
- Nếu cả hai accuracy đều thấp, mô hình có thể chưa học đủ hoặc cần điều chỉnh hyperparameter.

### 6.3. Confusion Matrix
**Hình 3.** Confusion matrix của MLP.  
[Chèn reports/mlp/confusion_matrix.png]

**Hình 4.** Confusion matrix của CNN.  
[Chèn reports/cnn/confusion_matrix.png]

Phân tích các lớp có nhiều dự đoán sai nhất và so sánh khả năng phân biệt các nhóm quần áo có hình dáng tương tự.

### 6.4. Ảnh dự đoán
**Hình 5.** Ví dụ dự đoán ngẫu nhiên của MLP.  
[Chèn reports/mlp/predictions_random.png]

**Hình 6.** Ví dụ dự đoán ngẫu nhiên của CNN.  
[Chèn reports/cnn/predictions_random.png]

**Hình 7.** Các ảnh dự đoán sai của hai mô hình.  
[Chèn predictions_wrong.png tương ứng]

Ảnh minh họa giúp kiểm tra trực quan nhãn dự đoán và nhãn thực tế, bổ sung cho các chỉ số tổng hợp.

## 7. THẢO LUẬN

### 7.1. So sánh chất lượng phân loại
Dựa trên test accuracy và macro-F1 thực tế, xác định mô hình có hiệu quả cao hơn. Không nên chỉ dựa vào accuracy vì macro-F1 phản ánh hiệu suất trung bình trên từng lớp.

### 7.2. So sánh chi phí tính toán
Đối chiếu thời gian huấn luyện, inference latency và số lượng tham số. MLP thường có cấu trúc đơn giản; CNN có thêm phép tích chập nhưng có khả năng tận dụng cấu trúc không gian của ảnh.

### 7.3. Phân tích lỗi
Dùng confusion matrix và ảnh dự đoán sai để nhận diện các cặp lớp thường bị nhầm. Những lớp có hình dáng tương tự có thể khó phân biệt ở độ phân giải 28×28.

## 8. KẾT LUẬN
Bài thực hành đã xây dựng pipeline PyTorch cho bài toán phân loại FashionMNIST, bao gồm tải dữ liệu, tiền xử lý, DataLoader, xây dựng mô hình, training loop, validation, lưu/nạp checkpoint và đánh giá. Hai kiến trúc được khảo sát là MLP và CNN.

Kết luận cuối cùng về mô hình tốt hơn cần được bổ sung sau khi hoàn thành các lần chạy thực nghiệm và điền số liệu ở Chương 6. Qua bài thực hành, người học có thể hiểu rõ hơn sự khác biệt giữa mạng fully connected và convolutional network khi xử lý ảnh.

## 9. HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN
- FashionMNIST có độ phân giải thấp, hạn chế chi tiết hình ảnh.
- Chưa khảo sát nhiều cấu hình kiến trúc và learning rate.
- Có thể bổ sung data augmentation, scheduler và thử nghiệm nhiều seed.
- Có thể đánh giá thêm precision/recall từng lớp và thời gian suy luận trên CPU/GPU.

## 10. TÀI LIỆU THAM KHẢO
1. Fashion-MNIST dataset: https://github.com/zalandoresearch/fashion-mnist
2. PyTorch Tutorials: https://pytorch.org/tutorials/
3. PyTorch Neural Network API: https://pytorch.org/docs/stable/nn.html
4. PyTorch Data Loading: https://pytorch.org/docs/stable/data.html
