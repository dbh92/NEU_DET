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
| `src/layers.py` | `Layer` (base: `forward(X, training)`, `backward(dout)`, `params()`, `grads()`), `Dense(in, out, init="he"\|"xavier", rng, dtype=np.float32)` — `W (in,out)`, `b (out,)`, `dW`, `db`, cache `self.X`. `xavier` = Glorot `sqrt(2/(in+out))`. `backward`: `dX = dout @ W.T`, `dW[...] = X.T @ dout`, `db[...] = dout.sum(0)` |
| `src/activations.py` | `ReLU` (lưu `self.mask = X > 0`, `backward = dout * mask`), `LeakyReLU(alpha)`, `softmax(Z)` (ổn định, trừ max) |
| `src/losses.py` | `log_softmax(Z)`, `cross_entropy(P, Y, eps)` (Cách A, bị chặn trần ở 27.63), `SoftmaxCrossEntropy` (`forward(Z, Y) -> float` nhận LOGITS, `backward() -> (P-Y)/B`), `accuracy(scores, y)`, `test_luong_tinh_loss()` |
| `src/optimizers.py` | `Optimizer` (base: gom tham chiếu params/grads một lần trong `__init__`, `step()`, `zero_grad()`), `SGD(layers, lr)`, `Momentum(layers, lr, momentum=0.9)`, `Adam(layers, lr=1e-3, beta1, beta2, eps, bias_correction=True)`. Mọi cập nhật phải **ghi tại chỗ** (`p -= ...`). lr đề xuất cho mạng 4096→128→6: SGD 0.1, Momentum 0.01, Adam 1e-3 |
| `src/model.py` | `Sequential(layers)` — `forward(X, training)` lặp xuôi, `backward(dout)` lặp **reversed**, `predict(X)`, `params()`, `buffers()`, `so_tham_so()`; `tao_mlp(in, hidden, C, rng, dropout=0.0, batchnorm=False)` → `Dense [-> BN] -> ReLU [-> Dropout] -> ... -> Dense`. Loss nằm NGOÀI Sequential |
| `src/train.py` | `danh_gia(...)` → `(loss, acc)`, `lat_ngau_nhien(Xb, rng, size)` (augment: lật ngang/dọc), `train(model, opt, crit, data, epochs, batch_size, rng, verbose, patience, khoi_phuc_tot_nhat, augment)` → `history` (4 list + `no_o_epoch`, `epoch_tot_nhat`, `val_acc_tot_nhat`, `dung_som_o_epoch`); khôi phục trọng số tốt nhất phải mang theo **cả `buffers()`**, `ve_duong_cong(...)`, `so_sanh(...)` |
| `src/conv.py` | `im2col(X, KH, KW, stride, pad)` → `(N*OH*OW, C*KH*KW)`, `col2im(...)` (CỘNG DỒN chỗ chồng lấn), `Conv2D(in_c, out_c, k, stride, padding, rng, dtype)` (W `(C_out,C_in,KH,KW)`, He với `fan_in = C_in*KH*KW`), `MaxPool2D(k, stride)` (gộp C vào N, nhớ `argmax`), `Flatten()`, `conv_ngay_tho(...)` để đối chiếu. Ảnh dạng **(N, C, H, W)** |
| `src/cnn_train.py` | `du_lieu_dang_anh(data)` (reshape `(N,4096)` → `(N,1,64,64)`), `ve_bo_loc(conv)` → `outputs/bo_loc_conv1.png`, `chay_cnn(...)`; main so sánh 4 cấu hình CNN + mốc MLP |
| `src/checkpoint.py` | `BANG_LOP` (bảng tra lớp, **không dùng pickle** vì pickle = chạy code tuỳ ý), `luu_model(model, path, meta)` / `nap_model(path)` → `.npz` chứa kiến trúc JSON + `p{i}.{ten}` + `b{i}.{ten}` + meta; `du_doan_anh(model, meta, duong_dan, top_k)`, `du_doan_nhieu_anh(...)`. meta **bắt buộc** có `mean`, `std`, `img_size`, `class_names` |
| `src/train_final.py` | `train_va_luu(epochs, seed, path)` — train CNN (16,32)+aug+BN rồi lưu `checkpoints/cnn_neu_det.npz`, kiểm chứng nạp lại khớp tuyệt đối |
| `src/predict.py` | CLI: `python src/predict.py <ảnh.jpg>`; không tham số thì lấy ảnh val làm ví dụ |
| `src/regularization.py` | `CAC_CAU_HINH` (9 cấu hình), `chay_mot_cau_hinh(...)`, `so_sanh_chong_overfit(data, epochs)`; main chạy bảng so sánh + đánh giá cấu hình tốt nhất → `outputs/*_chong_overfit.png` |
| `src/evaluate.py` | `confusion_matrix(y_true, y_pred, C)` (hàng=thật, cột=đoán; dùng `np.bincount`), `tinh_chi_so(cm)` → precision/recall/f1/support + macro (chia an toàn, lớp không được đoán → 0.0 chứ không nan), `in_confusion_matrix`, `in_bao_cao`, `top_k_accuracy(scores, y, k)`, `ve_confusion_matrix` → `outputs/confusion_matrix.png`, `xem_anh_sai` → `outputs/anh_sai.png` (sắp theo độ tự tin giảm dần), `danh_gia_day_du(...)` |
| `src/gradcheck.py` | `numerical_gradient(f, x, h=1e-4)` (sai phân trung tâm; h nhỏ hơn lại TỆ hơn do làm tròn), `rel_error`, `check_gradients(seed, nguong=1e-6)` (mạng 5→4→3 float64), `overfit_batch_nho()` |

## Tiến độ

- [x] **1.** Data loader + cache
- [x] **2.** Tiền xử lý + `show_samples`
- [x] **3.** `Layer`, `Dense` forward (thí nghiệm He init: std=1 → 4e10, std=0.01 → 3e-10, He → 0.87)
- [x] **4.** `ReLU`, `LeakyReLU`, `softmax` (forward pass đầu tiên: acc ≈ 1/6, ReLU zero ≈ 49%)
- [x] **5.** Cross-Entropy (`log_softmax`, `cross_entropy`, `SoftmaxCrossEntropy`, `accuracy`) — loss ban đầu ≈ 2.0–2.6 quanh ln6
- [x] **6.** Backpropagation + `src/gradcheck.py` (sai số ~1e-9, overfit 64 ảnh: loss 2.67 → 0.002, acc 1.0)
- [x] **7.** Optimizer: `SGD`, `Momentum`, `Adam` (thung lũng k=100: SGD 260 bước, Momentum 95, Adam 86)
- [x] **8.** Training loop (`src/model.py`, `src/train.py`) — MLP [128] + Adam 1e-3, 30 epoch: train acc **0.90** / val acc **0.48** (tốt nhất 0.53 ở epoch 14), chênh lệch **+0.42** → overfit đúng như dự đoán. SGD lr=0.1 NỔ ở epoch 11 trên toàn tập (dù học thuộc 64 ảnh rất tốt ở Bước 7)
- [x] **9.** Đánh giá (`src/evaluate.py`) — val: acc 0.481, top-2 0.689, top-3 0.903; train acc 0.943. Nhầm lẫn thật (KHÁC dự đoán ban đầu): `rolled-in_scale` → `scratches` 52% và → `inclusion` 45% (recall chỉ **0.033**), `crazing` → `pitted_surface` 40%. Mạng dồn dự đoán vào `pitted_surface` (104 lần / 60 ảnh thật) và gần như bỏ rơi `rolled-in_scale` (12 lần)
- [x] **10.** Chống overfit (`src/regularization.py`) — tốt nhất: `[256,128]` + BatchNorm + Dropout 0.5 + AdamW 1e-4 + lật ảnh: **val acc 0.611** (cơ sở 0.547), **val loss 0.876** (cơ sở 1.631), **chênh lệch +0.069** (cơ sở +0.421). Nhưng `rolled-in_scale` vẫn recall **0.000** (83% bị đoán thành `inclusion`)
- [x] **11.** CNN bằng NumPy (`src/conv.py`, `src/cnn_train.py`) — **val acc 0.9417** (MLP 0.6111), val loss **0.179** (MLP 0.929). `rolled-in_scale` recall **0.000 → 1.000**. im2col nhanh hơn vòng for 300 lần. Phần conv chỉ 4,800 tham số. ~10s/epoch với (16,32)
- [x] **12.** Lưu/nạp model (`src/checkpoint.py`, `src/train_final.py`, `src/predict.py`) — `checkpoints/cnn_neu_det.npz` 1.88 MB, nạp lại khớp tuyệt đối 0.9417. Đo được hai cái bẫy: quên buffer mất 0.069 accuracy; quên chuẩn hóa → đoán sai với xác suất 1.000

**LỘ TRÌNH 12 BƯỚC ĐÃ HOÀN TẤT.** Nếu người dùng muốn đi tiếp, các hướng gợi ý:
learning-rate schedule, BatchNorm2d, kiến trúc sâu hơn (VGG thu nhỏ), k-fold, augmentation
mạnh hơn (xoay/cắt/đổi sáng), hoặc dùng `annotations/` để làm object detection.

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
