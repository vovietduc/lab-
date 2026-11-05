# project_description.md – Lab 1: Phân loại ảnh FashionMNIST

## 1. Định nghĩa bài toán
Phân loại ảnh (multi-class classification, học có giám sát): với mỗi ảnh xám 28×28 của một món đồ thời trang, dự đoán nó thuộc 1 trong 10 lớp.

| Nhãn | Lớp | Nhãn | Lớp |
|---|---|---|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

## 2. Dữ liệu
- **Nguồn:** Fashion-MNIST (Zalando Research), tải qua `torchvision.datasets.FashionMNIST`.
- **Quy mô:** 60.000 ảnh train gốc + 10.000 ảnh test, mỗi lớp cân bằng.
- **Đầu vào:** tensor `float32` shape `(B, 1, 28, 28)`, đã chuẩn hóa z-score bằng mean/std của tập train.
- **Đầu ra:** logits shape `(B, 10)`; nhãn dự đoán = `argmax`; xác suất = `softmax`.
- **Thách thức:** các lớp áo (T-shirt/top, Shirt, Pullover, Coat) rất giống nhau; ảnh nhỏ, ít chi tiết.

## 3. Chiến lược chia tập
| Tập | Số ảnh | Tỉ lệ ≈ | Dùng để |
|---|---|---|---|
| Train | 50.000 | 71% | Huấn luyện |
| Validation | 10.000 | 14% | Chọn hyperparameter, Early Stopping, chọn checkpoint |
| Test | 10.000 | 14% | Đánh giá cuối cùng, **chỉ dùng một lần** |

- Train/Val chia ngẫu nhiên từ 60.000 ảnh train gốc, cố định theo `seed = 42`.
- Test là tập test chuẩn của FashionMNIST (cố định 10.000 ảnh), nên tỉ lệ thực tế gần với khung 60–70% / 15–20% / 15–20% chứ không trùng tuyệt đối.
- mean/std chuẩn hóa chỉ tính trên tập train để tránh rò rỉ dữ liệu.

## 4. Hệ thống chỉ số đánh giá
| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy | Tỉ lệ ảnh dự đoán đúng (phù hợp vì dữ liệu cân bằng) |
| Macro-F1 | Trung bình F1 của 10 lớp, cho thấy lớp nào yếu |
| Loss (Cross-Entropy) | Mức sai lệch xác suất dự đoán |
| Confusion matrix | Các cặp lớp bị nhầm lẫn |
| Latency (ms/ảnh) | Tốc độ suy luận |
| Peak VRAM (MB) | Bộ nhớ GPU tiêu thụ |

## 5. Benchmark Baseline
- **Baseline:** MLP thuần 2 lớp ẩn (256, 128) trên ảnh phẳng 784 chiều (`configs/mlp.yaml`).
- **Mô hình chính:** Backbone ResNet-18 + MLP Head (`configs/default.yaml`).
- **Tiêu chí:** mô hình chính phải được so sánh với baseline trên cùng tập val/test, cùng seed. Giá trị baseline thực tế được ghi ở bảng Results trong `README.md` sau khi chạy.
