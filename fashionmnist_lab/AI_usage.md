# AI_usage.md – Nhật ký sử dụng và kiểm duyệt AI

> Tài liệu này phải phản ánh **đúng thực tế** bạn đã làm. Các ô để trống (`___`) do bạn tự điền sau khi chạy và kiểm tra. Không điền số liệu bạn chưa kiểm chứng.

## 1. Tools Ecosystem
| Công cụ | Dùng cho việc gì |
|---|---|
| Claude (Anthropic) | Soạn khung dự án, mã nguồn nháp (`src/`, `tests/`), tài liệu (`*.md`), giải thích khái niệm |
| ___ (thêm công cụ khác nếu có: Copilot, Cursor, Gemini...) | ___ |

## 2. Prompting Framework
Công thức: **Prompt chuẩn = Context chuẩn + Phạm vi nhỏ + Tiêu chuẩn kiểm chứng**

| Thành phần | Áp dụng trong lab này |
|---|---|
| Context chuẩn | Bài toán FashionMNIST, PyTorch 2.x, máy có GPU, đề bài Practice 1 |
| Phạm vi nhỏ | Mỗi lần yêu cầu một module (dataset, model, train, evaluate...) |
| Tiêu chuẩn kiểm chứng | Mỗi file có khối `if __name__ == "__main__":` để Smoke Run; có `pytest` |

### Các prompt quan trọng (ghi lại prompt thật bạn đã dùng)
1. ___
2. ___
3. ___

## 3. Module Contribution Breakdown
Ước lượng trung thực tỉ lệ đóng góp trên từng file. Nên dựa trên việc bạn đã tự đọc, sửa, hiểu những gì.

| File | AI soạn nháp | Người chỉnh sửa / viết thêm | % AI | % Người |
|---|---|---|---|---|
| `src/dataset.py` | Có | ___ | ___ | ___ |
| `src/model.py` | Có | ___ | ___ | ___ |
| `src/train.py` | Có | ___ | ___ | ___ |
| `src/evaluate.py` | Có | ___ | ___ | ___ |
| `src/predict.py`, `src/utils.py` | Có | ___ | ___ | ___ |
| `tests/` | Có | ___ | ___ | ___ |
| `project_*.md`, `README.md` | Có | ___ | ___ | ___ |
| Notebook phân tích dữ liệu và thí nghiệm | Có | ___ | ___ | ___ |

## 4. Verification & Audit Log
Ghi lại kết quả **thật** của từng bước kiểm chứng.

| Bước | Lệnh | Kết quả / Lỗi gặp | Cách khắc phục |
|---|---|---|---|
| Smoke Run model | `python -m src.model` | ___ | ___ |
| Smoke Run dataset | `python -m src.dataset` | ___ | ___ |
| Dry Run | `python -m src.train --dry-run` | ___ | ___ |
| Normal Run | `python -m src.train` | ___ | ___ |
| Evaluate | `python -m src.evaluate` | ___ | ___ |
| Unit test | `pytest` | ___ | ___ |

### Checklist rà soát mã do AI sinh ra
- [ ] `model.eval()` khi validation/test, `model.train()` khi huấn luyện
- [ ] `torch.no_grad()` trong validation/test
- [ ] Không có phép in-place làm hỏng đồ thị autograd
- [ ] Shape khớp ở mọi lớp (đã Smoke Run)
- [ ] Không bị CUDA OOM (đã Dry Run, theo dõi VRAM)
- [ ] Tập test chỉ dùng để báo cáo cuối, không dùng để chọn tham số
- [ ] Tôi hiểu từng dòng code và có thể giải thích khi được hỏi
