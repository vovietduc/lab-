# FashionMNIST Data Audit

## Mục tiêu
Thực hiện kiểm tra kỹ thuật cơ bản đối với dữ liệu FashionMNIST trước khi huấn luyện MLP và CNN. Đây là audit chất lượng dữ liệu ở mức bài thực hành, không phải kiểm toán dữ liệu doanh nghiệp.

## Các nội dung
1. Số lượng: 60.000 ảnh train gốc và 10.000 ảnh test.
2. Kích thước: ảnh grayscale 28×28.
3. Kiểu dữ liệu: uint8; miền pixel 0–255.
4. Nhãn: các giá trị từ 0 đến 9.
5. Chia tập: 50.000 train và 10.000 validation, seed 42; chỉ số hai tập không giao nhau.
6. Phân phối lớp: cả 10 lớp phải xuất hiện trong mỗi split.
7. Chuẩn hóa: mean/std được tính từ train split; standard deviation phải dương và thống kê phải hữu hạn.

## Cách chạy
Đặt script tại `src/data_audit.py`, chạy từ thư mục `fashionmnist_lab`:

```powershell
python -m src.data_audit
```

Lần đầu chạy có thể tải FashionMNIST vào thư mục `./data`.

## Kiểm soát data leakage
Không dùng test set để tính mean/std, chọn checkpoint hoặc tinh chỉnh hyperparameter. Mean/std chỉ lấy từ train split; validation phục vụ chọn checkpoint; test chỉ dùng đánh giá cuối cùng.

## Trạng thái
Script mới cung cấp các phép kiểm tra và in kết quả PASS/FAIL. Chỉ ghi nhận audit đã đạt sau khi chạy script trên môi trường project thực tế.
