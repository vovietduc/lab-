# BÁO CÁO THỰC HÀNH – CHAPTER 2
## FASHIONMNIST IMAGE CLASSIFICATION: MLP VS CNN

**Học phần:** Artificial Neural Networks / Deep Learning  
**Bài thực hành:** Practice 1 – PyTorch FashionMNIST Classification  
**Công cụ:** Python, PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Scikit-learn  
**Dữ liệu:** FashionMNIST  
**Phạm vi:** MLP và CNN; không đưa ResNet18 vào so sánh.

---

## TÓM TẮT

Bài thực hành xây dựng pipeline phân loại ảnh FashionMNIST bằng PyTorch và so sánh hai kiến trúc MLP, CNN. Pipeline bao gồm khảo sát dữ liệu, tiền xử lý, chia train/validation/test, xây dựng mô hình, huấn luyện bằng backpropagation, chọn checkpoint theo validation accuracy và đánh giá trên test set.

Trong kết quả hiện có, MLP đạt test accuracy 88,34% và Macro-F1 88,35%; CNN đạt test accuracy 92,20% và Macro-F1 92,15%. CNN cao hơn 3,86 điểm phần trăm về accuracy và 3,80 điểm phần trăm về Macro-F1. CNN có chất lượng phân loại tốt hơn trong cấu hình thử nghiệm, trong khi MLP có thời gian huấn luyện và latency thấp hơn trong lần đo này.

## 1. GIỚI THIỆU

FashionMNIST là bài toán phân loại ảnh đa lớp gồm 10 nhóm quần áo và phụ kiện. Mục tiêu là thực hành quy trình học có giám sát bằng PyTorch, từ tải dữ liệu, tạo Tensor và DataLoader đến xây dựng mạng nơ-ron, tối ưu trọng số, đánh giá và lưu mô hình.

Hai cách tiếp cận được khảo sát:
- **MLP:** sử dụng các lớp fully connected trên vector ảnh đã Flatten.
- **CNN:** sử dụng convolution để học đặc trưng không gian từ ảnh.

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. MLP

MLP gồm input layer, hidden layers và output layer. Ảnh 28×28 được Flatten thành vector 784 chiều. Kiến trúc sử dụng Linear(784,256) → ReLU → Dropout(0,2) → Linear(256,128) → ReLU → Dropout(0,2) → Linear(128,10).

Ưu điểm là cấu trúc đơn giản, dễ triển khai và phù hợp làm baseline. Hạn chế là Flatten làm mất bố cục không gian hai chiều của ảnh.

### 2.2. CNN

CNN xử lý tensor ảnh 2D và dùng kernel convolution để học đặc trưng cục bộ. Mô hình có ba tầng convolution với 32, 64 và 128 kênh; kết hợp Batch Normalization, ReLU, Max Pooling, Adaptive Average Pooling và classifier 1152 → 128 → 10.

CNN giữ được quan hệ không gian giữa pixel và chia sẻ trọng số trong kernel, nhờ đó phù hợp hơn với dữ liệu ảnh.

### 2.3. Training loop và backpropagation

Với mỗi mini-batch, mô hình thực hiện:
1. Forward pass để tạo logits.
2. CrossEntropyLoss để đo sai số giữa logits và nhãn.
3. Backward pass bằng `loss.backward()` để tính gradient.
4. Optimizer update bằng `optimizer.step()`.
5. Xóa gradient cũ bằng `optimizer.zero_grad()`.

Sau mỗi epoch, validation được sử dụng để theo dõi khả năng tổng quát hóa và chọn checkpoint. Test set chỉ được dùng cho đánh giá cuối cùng.

## 3. DỮ LIỆU VÀ TIỀN XỬ LÝ

FashionMNIST có 70.000 ảnh grayscale kích thước 28×28, gồm 60.000 ảnh train gốc và 10.000 ảnh test. Mỗi lớp có 6.000 ảnh train và 1.000 ảnh test.

| Tập | Số ảnh | Mục đích |
|---|---:|---|
| Train | 50.000 | Cập nhật trọng số |
| Validation | 10.000 | Theo dõi, chọn checkpoint |
| Test | 10.000 | Đánh giá cuối cùng |

60.000 ảnh train gốc được chia train/validation bằng seed 42. Test set được giữ riêng. Pixel được chuyển về [0,1] và chuẩn hóa theo:

`x_norm = (x - mean_train) / std_train`

Mean/std chỉ được tính từ training split nhằm tránh data leakage. Thống kê audit là mean = 0,2858626 và std = 0,3529320. CNN nhận tensor (B,1,28,28); MLP Flatten thành vector 784 chiều.

## 4. PHÂN TÍCH DỮ LIỆU (EDA)

Notebook thực hiện khảo sát dữ liệu thô, cleaning, class distribution, trực quan ảnh, ảnh trung bình theo lớp, thống kê pixel, khoảng cách giữa ảnh trung bình các lớp và PCA.

### 4.1. Tổng quan và phân phối lớp

Dữ liệu gốc có 10 lớp cân bằng: mỗi lớp gồm 6.000 ảnh trong tập train và 1.000 ảnh trong tập test. Sau khi chia ngẫu nhiên train/validation với seed 42, số lượng từng lớp có dao động nhỏ giữa hai tập, nhưng cả 10 lớp đều xuất hiện trong từng split.

### 4.2. Cleaning

Kết quả đã lưu trong output notebook:
- Ảnh trống trong train: **0**.
- Ảnh trùng lặp chính xác trong train: **0**.
- Ảnh train xuất hiện chính xác trong test: **0**.

Kiểm tra trùng lặp sử dụng hash nội dung pixel. Kết quả này không loại trừ các ảnh gần giống về mặt ngữ nghĩa.

### 4.3. Thống kê pixel

Trên dữ liệu train thô sau khi đưa pixel về [0,1]:
- Mean toàn bộ: khoảng 0,2860.
- Standard deviation: khoảng 0,3530.
- Tỷ lệ pixel bằng 0: khoảng 50,21%.

Các giá trị pixel bằng 0 chiếm tỷ lệ lớn do nền ảnh tối. Điều này phản ánh đặc điểm ảnh grayscale của FashionMNIST.

### 4.4. Độ tương đồng giữa các lớp

Khoảng cách Euclid giữa ảnh trung bình của các lớp cho thấy một số cặp có hình dáng gần nhau:

| Cặp lớp | Khoảng cách |
|---|---:|
| Pullover – Coat | 2,48 |
| Pullover – Shirt | 2,68 |
| Coat – Shirt | 3,02 |
| T-shirt/top – Shirt | 3,72 |
| Sandal – Sneaker | 3,84 |

Các lớp trang phục phía trên có hình dạng tương đối tương đồng. Đây là một gợi ý để phân tích các trường hợp nhầm lẫn ở Confusion Matrix; khoảng cách ảnh trung bình không phải thước đo trực tiếp về độ khó phân loại.

### 4.5. PCA

PCA được thực hiện trên một mẫu ngẫu nhiên 6.000 ảnh. Với 50 thành phần chính, tỷ lệ phương sai được giữ lại khoảng **86,3%**. Kết quả giúp minh họa khả năng biểu diễn dữ liệu trong không gian chiều thấp; đây không phải bước thay thế đầu vào của hai mô hình cuối cùng.

## 5. THIẾT KẾ VÀ THIẾT LẬP THỰC NGHIỆM

| Đặc điểm | MLP | CNN |
|---|---|---|
| Input | Flatten 784 chiều | Ảnh 1×28×28 |
| Thành phần | Linear, ReLU, Dropout | Conv2D, BatchNorm, ReLU, Pooling |
| Kiến trúc | 784→256→128→10 | Conv 1→32→64→128; GAP; 1152→128→10 |
| Số tham số | 235.146 | 241.994 |

| Tham số huấn luyện | Giá trị |
|---|---:|
| Seed | 42 |
| Batch size | 128 |
| Epoch tối đa | 15 |
| Optimizer | Adam |
| Learning rate | 0,001 |
| Loss | CrossEntropyLoss |
| Weight decay | 0,0001 |
| Early stopping patience | 4 |
| MLP dropout | 0,2 |
| CNN dropout | 0,25 |

Hai mô hình dùng cùng cách chia dữ liệu và test set. Checkpoint được chọn theo validation accuracy tốt nhất. MLP chạy 15 epoch; CNN dừng sau 15 epoch do không cải thiện validation accuracy trong 4 epoch liên tiếp.

## 6. KẾT QUẢ THỰC NGHIỆM

### 6.1. So sánh tổng thể

| Chỉ số | MLP | CNN |
|---|---:|---:|
| Best validation accuracy | 89,15% | 92,25% |
| Test accuracy | 88,34% | 92,20% |
| Test Macro-F1 | 88,35% | 92,15% |
| Test loss | 0,3276 | 0,2331 |
| Training time | 15 s | 250 s |
| Inference latency | 0,067 ms/ảnh | 0,339 ms/ảnh |
| Parameters | 235.146 | 241.994 |
| Epoch thực chạy | 15 | 15 |

CNN cao hơn MLP 3,86 điểm phần trăm về test accuracy và 3,80 điểm phần trăm về Macro-F1. Test loss thấp hơn cho thấy dự đoán của CNN có chất lượng tốt hơn theo hàm mất mát đang sử dụng.

MLP có thời gian train và latency thấp hơn trong lần đo hiện tại. Dù số tham số hai mô hình gần nhau, CNN có phép tích chập trên feature maps nên chi phí tính toán thực tế không thể suy ra chỉ từ số tham số.

### 6.2. Đường cong huấn luyện

Bốn biểu đồ so sánh train loss, validation loss, train accuracy và validation accuracy được tạo từ `history.json` của từng mô hình. CNN có 15 epoch thực chạy, MLP có 15 epoch; không nội suy thêm điểm cho CNN.

- Train loss giảm ở cả hai mô hình.
- Validation loss có dao động trong quá trình huấn luyện.
- CNN đạt validation loss thấp nhất khoảng 0,2296 tại epoch 6.
- CNN đạt validation accuracy tốt nhất 92,25% tại epoch 8; checkpoint tốt nhất được khôi phục từ epoch này.
- MLP đạt validation accuracy tốt nhất 89,15% tại epoch 15.

### 6.3. Phân tích theo từng lớp

| Lớp | MLP F1 | CNN F1 | Nhận xét |
|---|---:|---:|---|
| T-shirt/top | 0,8387 | 0,8801 | CNN tốt hơn |
| Trouser | 0,9786 | 0,9890 | Cả hai tốt |
| Pullover | 0,7978 | 0,8865 | CNN cải thiện rõ |
| Dress | 0,8892 | 0,9235 | CNN tốt hơn |
| Coat | 0,8047 | 0,8714 | CNN tốt hơn |
| Sandal | 0,9599 | 0,9800 | CNN tốt hơn |
| Shirt | 0,7017 | 0,7738 | Lớp khó nhất |
| Sneaker | 0,9402 | 0,9578 | CNN tốt hơn |
| Bag | 0,9701 | 0,9881 | MLP nhỉnh hơn |
| Ankle boot | 0,9537 | 0,9645 | CNN tốt hơn |

CNN đạt F1 cao hơn ở 9/10 lớp. Shirt là lớp có F1 thấp nhất ở cả hai mô hình; Bag là lớp duy nhất MLP nhỉnh hơn CNN.

## 7. CONFUSION MATRIX VÀ PHÂN TÍCH LỖI

Confusion Matrix được tạo từ checkpoint hiện tại và 10.000 ảnh test. Hàng biểu diễn lớp thực tế, cột biểu diễn lớp dự đoán. Phần tử trên đường chéo là dự đoán đúng; phần tử ngoài đường chéo là số ảnh bị nhầm.

### 7.1. MLP

![MLP Confusion Matrix](../mlp/confusion_matrix.png)

MLP dự đoán đúng 8.834/10.000 ảnh, tương ứng 88,34%. Các cặp nhầm lẫn lớn nhất:

| Lớp thực tế | Dự đoán thành | Số ảnh |
|---|---|---:|
| Shirt | T-shirt/top | 156 |
| Pullover | Shirt | 87 |
| Pullover | Coat | 84 |
| Coat | Pullover | 83 |
| Coat | Shirt | 77 |

Các nhầm lẫn chủ yếu tập trung ở nhóm áo. Điều này phù hợp với EDA về khoảng cách ảnh trung bình giữa Pullover, Coat, Shirt và T-shirt/top.

### 7.2. CNN

![CNN Confusion Matrix](../cnn/confusion_matrix.png)

CNN dự đoán đúng 9.220/10.000 ảnh, tương ứng 92,20%. Các cặp nhầm lẫn lớn nhất:

| Lớp thực tế | Dự đoán thành | Số ảnh |
|---|---|---:|
| Coat | Shirt | 95 |
| T-shirt/top | Shirt | 75 |
| Pullover | Shirt | 66 |
| Coat | Pullover | 63 |
| Dress | Shirt | 51 |

CNN giảm tổng số lỗi từ 1.166 xuống 780 ảnh. Một số cặp lỗi giảm rõ:
- Pullover → Coat: 84 xuống 36 ảnh.
- Shirt → T-shirt/top: 156 xuống 105 ảnh.
- Pullover → Shirt: 87 xuống 66 ảnh.

Tuy nhiên, Coat → Shirt tăng từ 77 lên 95 ảnh. CNN cải thiện chất lượng tổng thể nhưng không làm giảm mọi dạng nhầm lẫn.

### 7.3. Kết luận từ Confusion Matrix

Hai mô hình đều nhận diện tốt Trouser, Sandal và Bag. Các trường hợp nhầm lẫn tập trung nhiều hơn ở nhóm áo có hình dáng tương tự. CNN cải thiện rõ số ảnh nhận diện đúng của lớp Shirt: từ 682 lên 775 ảnh. Tuy vậy, Shirt vẫn là lớp khó nhất và tiếp tục gây nhầm lẫn với T-shirt/top, Pullover, Coat và Dress.

## 8. DATA AUDIT

File audit đã lưu tại `reports/data_audit/audit_summary.json` và `audit_log.txt`, ghi nhận **12/12 checks PASS**.

Các nhóm kiểm tra:
1. Số lượng 60.000 ảnh train gốc và 10.000 ảnh test.
2. Shape ảnh 28×28.
3. Grayscale, dtype uint8.
4. Miền pixel [0,255].
5. Nhãn hợp lệ [0,9].
6. Đủ 10 lớp trong dữ liệu.
7. Train và validation không giao nhau.
8. Kích thước split chính xác.
9. Train và validation bao phủ toàn bộ tập train gốc.
10. Đủ 10 lớp trong từng split.
11. Mean/std chuẩn hóa hữu hạn.
12. Standard deviation dương.

Mean = 0,2858626 và std = 0,3529320, được tính từ training split.

Ngoài 12 checks trên, notebook EDA đã kiểm tra ảnh trống, ảnh trùng lặp chính xác trong train và trùng lặp chính xác giữa train/test; kết quả ghi nhận 0 ở cả ba mục. Đây là các kiểm tra cleaning trong notebook, không thuộc 12 checks của script audit.

## 9. THẢO LUẬN VÀ HẠN CHẾ

### 9.1. Chất lượng phân loại

CNN có test accuracy và Macro-F1 cao hơn trong cấu hình thực nghiệm. Lợi thế này phù hợp với khả năng khai thác đặc trưng không gian của convolution. Kết luận chỉ áp dụng cho cấu hình và lần chạy hiện tại, không khẳng định CNN luôn vượt mọi MLP.

### 9.2. Chi phí tính toán

MLP train trong 15 giây và latency 0,067 ms/ảnh; CNN tương ứng 250 giây và 0,339 ms/ảnh. CNN chậm hơn trong lần đo này dù số tham số chỉ chênh lệch 6.848. Thời gian phụ thuộc thiết bị, tải hệ thống, batch size và điều kiện đo.

### 9.3. Hạn chế

- Thực nghiệm dùng một seed và một cấu hình chính cho mỗi kiến trúc.
- Chưa báo cáo trung bình và độ lệch chuẩn trên nhiều lần chạy.
- Latency phụ thuộc môi trường và số lần warm-up.
- PCA được thực hiện trên mẫu 6.000 ảnh để khảo sát, không phải toàn bộ dataset.
- Các lớp trang phục tương tự vẫn gây nhầm lẫn.

## 10. KẾT LUẬN

Bài thực hành đã triển khai pipeline FashionMNIST bằng PyTorch: tải dữ liệu, EDA, tiền xử lý, Dataset/DataLoader, xây dựng MLP/CNN, forward, CrossEntropyLoss, backpropagation, Adam, validation, lưu/nạp checkpoint và đánh giá test.

Trong kết quả hiện có, CNN đạt test accuracy 92,20% và Macro-F1 92,15%, cao hơn MLP lần lượt 88,34% và 88,35%. CNN phù hợp hơn về chất lượng phân loại trong cấu hình thử nghiệm; MLP có lợi thế về thời gian huấn luyện và độ trễ suy luận. Phân tích Confusion Matrix cho thấy các lỗi còn tập trung ở nhóm áo có hình dáng tương tự, đặc biệt là Shirt.

## 11. HƯỚNG PHÁT TRIỂN

- Thử nhiều seed và báo cáo trung bình ± độ lệch chuẩn.
- Thử learning-rate scheduler và data augmentation.
- Mở rộng phân tích các ảnh dự đoán sai.
- Đo latency có kiểm soát trên CPU/GPU.
- Thử điều chỉnh kiến trúc để cải thiện lớp Shirt.

## 12. TÀI LIỆU THAM KHẢO

1. https://github.com/zalandoresearch/fashion-mnist
2. https://pytorch.org/tutorials/
3. https://pytorch.org/docs/stable/nn.html
4. https://pytorch.org/docs/stable/data.html
5. https://scikit-learn.org/stable/modules/model_evaluation.html
