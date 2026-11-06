# FashionMNIST Data Audit

## Mục tiêu
Kiểm tra chất lượng kỹ thuật của FashionMNIST trước khi huấn luyện MLP và CNN. Audit này tập trung vào số lượng, định dạng, nhãn, chia tập và thống kê chuẩn hóa; không phải kiểm toán dữ liệu doanh nghiệp.

## Cách chạy

Từ thư mục gốc `fashionmnist_lab`:

```powershell
python -m src.data_audit
```

Có thể tùy chỉnh:

```powershell
python -m src.data_audit --data-root ./data --val-size 10000 --seed 42
```

## Các kiểm tra trong script

Script `src/data_audit.py` thực hiện 12 checks:

1. Có 60.000 ảnh train gốc.
2. Có 10.000 ảnh test.
3. Ảnh có kích thước 28×28.
4. Ảnh grayscale dạng uint8.
5. Pixel nằm trong [0,255].
6. Nhãn nằm trong [0,9].
7. Đủ 10 lớp trong train gốc và test.
8. Train và validation không giao nhau.
9. Kích thước train/validation đúng theo `val_size`.
10. Hai split bao phủ toàn bộ train gốc.
11. Đủ 10 lớp trong từng split.
12. Mean/std hữu hạn và std dương.

Mean/std chỉ tính từ training split để tránh data leakage.

## Kết quả đã lưu trong repository

Tại thời điểm rà soát, `reports/data_audit/audit_summary.json` và `audit_log.txt` ghi nhận:

- **12/12 checks PASS**.
- Train gốc: 60.000; train split: 50.000.
- Validation: 10.000; test: 10.000.
- Image shape: 28×28.
- Mean: 0,2858625948.
- Std: 0,3529320061.
- Mean/std được tính từ training split only.

Các file đầu ra:
- `audit_summary.json`: kết quả có cấu trúc JSON.
- `audit_log.txt`: log PASS/FAIL.
- `class_distribution.csv`: phân phối lớp theo split.
- `class_distribution.png`: biểu đồ phân phối lớp.

## Cleaning EDA bổ sung trong notebook

Notebook `notebooks/01_data_analysis_and_experiments.ipynb` thực hiện các kiểm tra bổ sung không nằm trong 12 checks của script audit:
- Ảnh trống (toàn pixel 0) trong train.
- Ảnh trùng lặp chính xác trong train theo hash nội dung.
- Ảnh train trùng chính xác với test.

Output notebook hiện ghi nhận 0 ở cả ba kiểm tra này. Cần phân biệt kiểm tra trùng nội dung pixel với kiểm tra ảnh gần giống về mặt ngữ nghĩa.

## Kiểm soát data leakage

- Mean/std chỉ lấy từ train split.
- Validation dùng để theo dõi và chọn checkpoint.
- Test không dùng để tính thống kê chuẩn hóa, chọn checkpoint hoặc tinh chỉnh hyperparameter.
