# NEU-DET — Phân loại lỗi bề mặt thép cán nóng (thuần NumPy)

Bài tập luyện tập **Neural Network viết tay bằng NumPy**, không dùng bất kỳ
deep learning framework nào (không PyTorch, không TensorFlow/Keras, không scikit-learn).

## Bài toán

Phân loại ảnh lỗi bề mặt thép cán nóng thành 6 lớp:

| # | Lớp | Mô tả |
|---|-----|-------|
| 0 | `crazing`         | nứt chân chim |
| 1 | `inclusion`       | tạp chất lẫn vào |
| 2 | `patches`         | mảng loang |
| 3 | `pitted_surface`  | rỗ bề mặt |
| 4 | `rolled-in_scale` | vảy cán ép vào |
| 5 | `scratches`       | vết xước |

## Dữ liệu

Bộ **NEU-DET** (Northeastern University Surface Defect Database).
Ảnh xám `200 x 200`, định dạng JPEG.

| Tập | Số ảnh | Phân bố |
|-----|--------|---------|
| train      | 1440 | 240 ảnh / lớp (cân bằng) |
| validation |  360 |  60 ảnh / lớp (cân bằng) |

Thư mục `annotations/` chứa bounding box dạng Pascal VOC XML — dành cho bài toán
*object detection*, **không dùng** trong bài phân loại này. Nhãn được lấy từ tên
thư mục con trong `images/`.

## Cấu trúc dự án

```
neu_det/
├── datasets/NEU-DET/       # dữ liệu gốc
│   ├── train/{images,annotations}
│   └── validation/{images,annotations}
├── src/                    # mã nguồn
│   └── data_loader.py      # đọc JPEG -> numpy, cache .npz
├── cache/                  # file .npz sinh ra khi chạy (git bỏ qua)
└── neu_det_env/            # môi trường ảo (git bỏ qua)
```

## Môi trường

Python 3.10.11

| Thư viện | Vai trò |
|----------|---------|
| `numpy`      | **toàn bộ** phép toán của mạng nơ-ron |
| `Pillow`     | chỉ để giải nén file JPEG thành mảng số |
| `matplotlib` | chỉ để vẽ biểu đồ |
| `tqdm`       | thanh tiến trình |

Chạy:

```powershell
.\neu_det_env\Scripts\Activate.ps1
python src\data_loader.py
```

## Lộ trình

- [ ] **1.** Data loader: JPEG → numpy array, resize, cache
- [ ] **2.** Tiền xử lý: normalize, one-hot, shuffle
- [ ] **3.** Lớp `Dense` — forward
- [ ] **4.** Activation: ReLU, Softmax
- [ ] **5.** Loss: Cross-Entropy
- [ ] **6.** Backpropagation
- [ ] **7.** Optimizer: SGD → Momentum → Adam
- [ ] **8.** Training loop + mini-batch
- [ ] **9.** Đánh giá: accuracy, confusion matrix
- [ ] **10.** Chống overfit: L2, Dropout, BatchNorm
- [ ] **11.** CNN bằng NumPy (im2col)
- [ ] **12.** Lưu/nạp model, dự đoán ảnh mới
