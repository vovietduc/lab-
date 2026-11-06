# Tài liệu ôn vấn đáp – FashionMNIST MLP vs CNN

Tài liệu bám theo source hiện tại trong `src/`, cấu hình và kết quả lưu trong `reports/`.

## 1. Bài toán và dữ liệu

**Bài toán:** phân loại ảnh đa lớp có giám sát. Ảnh grayscale 28×28 là đầu vào, mô hình dự đoán một trong 10 lớp FashionMNIST.

**Dataset:** 70.000 ảnh: 60.000 train gốc và 10.000 test; mỗi lớp có 6.000 ảnh train và 1.000 ảnh test.

**Chia tập:** 50.000 train, 10.000 validation, 10.000 test. Seed 42 cố định phép chia. Train cập nhật trọng số; validation chọn checkpoint; test chỉ đánh giá cuối cùng.

**EDA đã làm:** kiểm tra nhãn/shape/dtype, ảnh trống, ảnh trùng lặp, phân phối lớp, ảnh mẫu, ảnh trung bình từng lớp, pixel statistics, khoảng cách Euclid giữa ảnh trung bình và PCA.

**Cleaning:** notebook ghi nhận 0 ảnh trống, 0 ảnh trùng chính xác trong train, 0 ảnh train trùng chính xác với test. Hash pixel không phát hiện ảnh gần giống ngữ nghĩa.

**PCA:** dùng mẫu 6.000 ảnh; 50 thành phần giữ khoảng 86,3% phương sai. PCA chỉ phục vụ EDA, không phải input model.

## 2. Tiền xử lý

Pixel được chia 255 để về [0,1], sau đó chuẩn hóa: `(x - mean) / std`.

Mean = 0,2858626 và std = 0,3529320, tính chỉ trên training split để tránh data leakage. Cùng thống kê đó được áp dụng cho train, validation và test.

CNN nhận batch shape `(B, 1, 28, 28)`, float32. Nhãn có shape `(B,)`, int64. MLP Flatten ảnh thành 784 features.

Dataset biểu diễn dữ liệu và cách lấy mẫu; DataLoader tạo mini-batch, shuffle và lặp qua dữ liệu. Train dùng shuffle=True; validation/test shuffle=False.

## 3. MLP và CNN

**MLP:** Flatten 784 → Linear 256 → ReLU → Dropout 0,2 → Linear 128 → ReLU → Dropout 0,2 → Linear 10. Có 235.146 tham số. Ưu điểm đơn giản; hạn chế là Flatten làm mất cấu trúc không gian 2D.

**CNN:** Conv 1→32→64→128, BatchNorm và ReLU; hai khối đầu có MaxPool; AdaptiveAvgPool về 3×3; Flatten; Linear 1152→128→10 với ReLU và Dropout 0,25. Có 241.994 tham số.

- ReLU tạo phi tuyến.
- Dropout ngẫu nhiên tắt neuron khi train để regularization; eval tự tắt Dropout.
- BatchNorm chuẩn hóa activation theo mini-batch khi train và dùng running statistics khi eval.
- MaxPool giảm kích thước feature map bằng max; AdaptiveAvgPool tạo kích thước đầu ra cố định 3×3.
- Logits là điểm số thô trước softmax. CrossEntropyLoss nhận logits trực tiếp, không cần softmax trước.

## 4. Training loop

Trong `src.train.run_epoch`:
1. `optimizer.zero_grad()`: xóa gradient cũ.
2. Forward: model tạo logits.
3. CrossEntropyLoss: tính loss.
4. `loss.backward()`: backpropagation tính gradient.
5. `optimizer.step()`: cập nhật trọng số.

Adam learning rate 0,001, weight decay 0,0001, batch size 128, tối đa 15 epoch, patience 4. Early stopping theo validation accuracy; khi không cải thiện 4 epoch liên tiếp thì dừng. Code lưu state_dict tốt nhất rồi nạp lại trước khi lưu checkpoint.

CNN chạy 12 epoch vì early stopping; MLP chạy đủ 15 epoch. AMP chỉ bật khi cấu hình cho phép và device là CUDA.

## 5. Metrics và Confusion Matrix

- Accuracy = số dự đoán đúng / tổng số mẫu.
- Precision: trong số mẫu dự đoán thuộc lớp, tỷ lệ thực sự thuộc lớp.
- Recall: trong số mẫu thật thuộc lớp, tỷ lệ được dự đoán đúng.
- F1 = trung bình điều hòa precision và recall.
- Macro-F1: trung bình F1 của 10 lớp, mỗi lớp trọng số như nhau.
- Confusion Matrix: hàng là nhãn thật, cột là nhãn dự đoán; đường chéo là đúng, ngoài đường chéo là nhầm.

## 6. Kết quả hiện tại

| Chỉ số | MLP | CNN |
|---|---:|---:|
| Best validation accuracy | 89,48% | 92,05% |
| Test accuracy | 88,65% | 91,30% |
| Test Macro-F1 | 88,62% | 91,32% |
| Test loss | 0,3227 | 0,2473 |
| Training time | 21,88 s | 201,26 s |
| Latency | 0,0673 ms/ảnh | 0,2684 ms/ảnh |
| Parameters | 235.146 | 241.994 |
| Epochs | 15 | 12 |

CNN hơn MLP 2,65 điểm phần trăm accuracy và 2,71 điểm phần trăm Macro-F1. MLP nhanh hơn trong lần đo này. Số tham số gần nhau không đồng nghĩa chi phí tính toán giống nhau; convolution tính trên nhiều vị trí không gian.

Shirt là lớp khó nhất: F1 MLP 0,6991; CNN 0,7506. CNN có F1 tốt hơn ở 9/10 lớp; Bag là ngoại lệ.

## 7. Audit, checkpoint và notebook

Script `src.data_audit` hiện ghi nhận 12/12 checks PASS: số lượng, shape, grayscale/uint8, miền pixel, nhãn, đủ lớp, split disjoint, kích thước split, bao phủ tập train gốc, đủ lớp mỗi split, mean/std hữu hạn và std dương.

Checkpoint lưu `state_dict`, config, mean, std và class names. Predict phải dùng đúng mean/std đã train.

Dry-run là smoke test 1 epoch trên 50 train, 20 validation, 20 test; không dùng số liệu dry-run trong báo cáo.

Notebook hiện gọi `src.train` và `src.evaluate` qua subprocess; đọc history/metrics từ `reports/mlp` và `reports/cnn`. Không tự định nghĩa lại model hoặc training loop. Cần mở notebook từ thư mục gốc project để `Path.cwd()` trỏ đúng root.

## 8. Câu hỏi giảng viên có thể hỏi

**Tại sao cần validation?** Chọn checkpoint và theo dõi tổng quát hóa mà không đụng test.

**Tại sao mean/std chỉ tính từ train?** Nếu dùng val/test, thông tin từ tập đánh giá lọt vào preprocessing, gây data leakage.

**Tại sao không softmax trước CrossEntropyLoss?** CrossEntropyLoss đã kết hợp log-softmax và negative log likelihood.

**Train accuracy tăng nhưng val accuracy giảm?** Có thể overfitting; xem validation loss, regularization, augmentation và early stopping.

**Nếu dữ liệu mất cân bằng?** Accuracy có thể gây hiểu lầm; xem per-class precision/recall/F1, Macro-F1 và cân nhắc class weights/sampling.

**CNN chính xác hơn nhưng chậm hơn vì sao?** Convolution thực hiện phép tính trên feature maps; số tham số không phản ánh toàn bộ FLOPs.

**Vì sao chọn checkpoint bằng validation accuracy nhưng báo cáo Macro-F1?** Code hiện tại chọn checkpoint theo validation accuracy; test metrics sau đó báo cáo cả accuracy, Macro-F1, loss và classification report. Đây là hai vai trò khác nhau.

**Hạn chế thực nghiệm?** Một seed/cấu hình cho mỗi mô hình; chưa có trung bình ± độ lệch chuẩn nhiều lần chạy; latency phụ thuộc phần cứng; PCA chỉ chạy trên mẫu 6.000 ảnh.

## 9. Checklist trước vấn đáp

- [ ] Chạy `python -m src.data_audit`
- [ ] Chạy `python -m src.train --dry-run`
- [ ] Chạy `pytest -q`
- [ ] Giải thích `dataset.py`, `model.py`, `train.py`, `evaluate.py`
- [ ] Phân biệt train/validation/test
- [ ] Giải thích normalization và data leakage
- [ ] Đọc Confusion Matrix, giải thích lớp Shirt
- [ ] Nhớ accuracy, Macro-F1, parameters và thời gian
- [ ] Giải thích vì sao CNN tốt hơn về metric nhưng chậm hơn
