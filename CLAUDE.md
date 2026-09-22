# CLAUDE.md — Ngữ cảnh dự án cho Claude Code

File này được Claude Code tự động nạp khi mở dự án. Nó thay cho lịch sử hội thoại:
đọc xong file này là đủ để tiếp tục hướng dẫn đúng chỗ đang dừng.

## Dự án

Bài tập luyện **Neural Network viết tay bằng NumPy** — phân loại 6 loại lỗi bề mặt thép
cán nóng (bộ NEU-DET). Người học là người dùng; Claude đóng vai **người hướng dẫn**.

## Cách làm việc (người dùng đã yêu cầu — BẮT BUỘC tuân theo)

- Trả lời bằng **tiếng Việt**.
- Hướng dẫn **từng bước**, chi tiết, dễ hiểu. Đầu mỗi bước đưa **đặc tả cụ thể**: tên file,
  tên class/method, chữ ký hàm, **input/output (shape, dtype)**, thuật toán gợi ý, bẫy hay gặp,
  bộ test trong `if __name__ == "__main__":`, **tiêu chí nghiệm thu** (output mong đợi),
  và 2–3 câu hỏi suy nghĩ.
- **Đợi người dùng code xong** mới sang bước kế tiếp. Không tự viết code của bước
  trừ khi người dùng bảo "làm giúp / sửa giúp".
- Khi người dùng nộp code: **chạy thật** để review (bảng kết quả, ✅ đúng / ❌ cần sửa /
  ⚠️ nhỏ), trả lời các câu hỏi suy nghĩ của bước trước, rồi mới đưa đặc tả bước sau.
- Commit/push chỉ khi người dùng yêu cầu.

## Luật chơi kỹ thuật

- **Chỉ NumPy** cho mọi phép toán của mạng. KHÔNG dùng torch / tensorflow / keras /
  sklearn / scipy.
- `cv2.imread` (người dùng chọn) chỉ để giải nén JPEG. Không dùng hàm xử lý nào khác của cv2.
- `matplotlib` chỉ để vẽ. `tqdm` cho thanh tiến trình.
- Mọi mảng dùng **float32**. Tham số và gradient phải **ghi tại chỗ** (`self.dW[...] = ...`),
  không gán lại, vì optimizer giữ tham chiếu qua `params()` / `grads()`.
- Đường dẫn tính theo `PROJECT_ROOT` (trong `data_loader.py`), không phụ thuộc cwd.

## Môi trường

- Python 3.10.11, venv `neu_det_env/` (không commit). Chạy: `neu_det_env\Scripts\python.exe src\<file>.py`
- Clone về máy mới:
  ```powershell
  python -m venv neu_det_env
  .\neu_det_env\Scripts\pip install -r requirements.txt
  ```
- Khi test bằng Bash: đặt `PYTHONIOENCODING=utf-8` (in tiếng Việt), `MPLBACKEND=Agg` (không mở cửa sổ).

## Dữ liệu

`datasets/NEU-DET/{train,validation}/images/<tên lớp>/*.jpg` — ảnh xám 200×200.
Train 1440 (240/lớp), val 360 (60/lớp), cân bằng. `annotations/` là bbox cho detection — bỏ qua.
Nhãn: `CLASS_NAMES = [crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches]` → 0..5.

⚠️ Dữ liệu load **theo thứ tự lớp**: `y_train[:64]` toàn là lớp 0. Mọi test lấy batch con
phải chọn chỉ số ngẫu nhiên (`rng.choice(N, 64, replace=False)`).

## Mã nguồn hiện có (API)

| File | Nội dung |
|---|---|
| `src/data_loader.py` | `CLASS_NAMES`, `IMG_SIZE=64`, `PROJECT_ROOT`, `DATA_ROOT`, `CACHE_PATH`; `resize_nearest(image, size)`, `load_image(path, size)`, `load_split(split_dir, size) -> (X uint8 (N,64,64), y int64)`, `build_cache()`, `load_cache() -> (X_train, y_train, X_val, y_val)` → `cache/neu_det_64.npz` |
| `src/preprocess.py` | `flatten`, `fit_standardizer(X) -> (mean, std)` (scalar, chỉ fit trên train), `apply_standardizer`, `one_hot(y, C)`, `shuffle_data(X, y, rng)`, `iterate_minibatches(X, y, batch_size, shuffle, rng)` (generator), `prepare_data() -> dict` (X_train (1440,4096) f32, Y_train one-hot, y_train, X_val, Y_val, y_val, mean≈128.80, std≈53.13), `show_samples(...)` → `outputs/samples.png` |
| `src/layers.py` | `Layer` (base: `forward(X, training)`, `backward(dout)`, `params()`, `grads()`), `Dense(in, out, init="he"\|"xavier", rng)` — `W (in,out)`, `b (out,)`, `dW`, `db`, cache `self.X`. `xavier` = Glorot `sqrt(2/(in+out))`. `backward` chưa viết. |
| `src/activations.py` | `ReLU` (lưu `self.mask = X > 0`), `LeakyReLU(alpha)`, `softmax(Z)` (ổn định, trừ max). `backward` chưa viết. |
| `src/losses.py` | **Rỗng** — Bước 5 |

## Tiến độ

- [x] **1.** Data loader + cache
- [x] **2.** Tiền xử lý + `show_samples`
- [x] **3.** `Layer`, `Dense` forward (thí nghiệm He init: std=1 → 4e10, std=0.01 → 3e-10, He → 0.87)
- [x] **4.** `ReLU`, `LeakyReLU`, `softmax` (forward pass đầu tiên: acc ≈ 1/6, ReLU zero ≈ 49%)
- [ ] **5.** Cross-Entropy loss ← **ĐANG Ở ĐÂY** (đặc tả đã đưa, người dùng chưa code)
- [ ] 6. Backpropagation (Dense, ReLU, SoftmaxCE: `dZ = (P - Y) / B`) + gradient check số học
- [ ] 7. Optimizer: SGD → Momentum → Adam
- [ ] 8. Training loop + mini-batch (dự đoán: overfit mạnh, 524k tham số / 1440 ảnh)
- [ ] 9. Đánh giá: accuracy, confusion matrix (dự đoán nhầm crazing ↔ rolled-in_scale)
- [ ] 10. Chống overfit: L2, Dropout, BatchNorm
- [ ] 11. CNN bằng NumPy (im2col)
- [ ] 12. Lưu/nạp model, dự đoán ảnh mới

### Việc còn treo từ review Bước 4 (góp ý, người dùng chưa sửa)
- Test 5 trong `activations.py` dùng `X_train[:64]` (toàn lớp 0) → đổi sang batch ngẫu nhiên.
- Bỏ `try/except ImportError` quanh `from preprocess import prepare_data` (che lỗi thật).
- Thông báo lỗi `LeakyReLU` ghi `[0, 1]` nhưng điều kiện là `[0, 1)`; vài lỗi chính tả.

## Đặc tả Bước 5 (đã đưa cho người dùng) — `src/losses.py`

- `log_softmax(Z) -> (B,C)`: `Z_shift = Z - max`; `Z_shift - log(sum(exp(Z_shift)))` (log-sum-exp).
- `cross_entropy(P, Y, eps=1e-12) -> float`: Cách A, `-sum(Y * log(clip(P, eps, 1))) / B`; kiểm tra shape.
- `class SoftmaxCrossEntropy` (không kế thừa Layer): `forward(Z, Y) -> float` nhận **logits**,
  lưu `self.P = softmax(Z)`, `self.Y`; loss = `-sum(Y * log_softmax(Z)) / B`. `backward()` → Bước 6.
- `accuracy(scores, y) -> float`.
- Tests: tính tay `Z=[[1,2,3]]` → lớp 2: 0.4076, lớp 0: 2.4076; `Z=0` → ln6 = 1.7918;
  `Z=[[0,0,100]]` sai tự tin: Cách A 27.63 (bị chặn) vs Cách B 100; A≈B với logits thường;
  loss ban đầu mạng thật (batch ngẫu nhiên 256) ≈ 1.8–2.3.
- Câu hỏi: (1) loss bị chặn trần thì gradient ra sao; (2) vì sao `/B` thay vì tổng;
  (3) loss ban đầu = 15 thì nghi chỗ nào.
