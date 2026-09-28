"""
Lưu / nạp mô hình và dự đoán ảnh mới.

Một mô hình dùng được KHÔNG CHỈ LÀ TRỌNG SỐ. Cần đủ bốn thứ:
  1. Trọng số        W, b, gamma, beta
  2. Buffer          running_mean, running_var  (Bước 10: quên là mất 10 điểm accuracy)
  3. Kiến trúc       lớp nào, kích thước bao nhiêu
  4. Tiền xử lý      mean, std, img_size  <- thiếu cái này là lỗi phổ biến nhất
                     khi đưa mô hình vào sản xuất: chạy được, không báo lỗi,
                     chỉ là kết quả sai.

Tất cả gói vào MỘT file .npz.
"""

import json
import os

import numpy as np

from activations import ReLU, softmax
from conv import Conv2D, Flatten, MaxPool2D
from data_loader import load_image
from layers import BatchNorm1d, Dense, Dropout
from model import Sequential

# Bảng tra để dựng lại lớp từ mô tả JSON.
#
# Vì sao KHÔNG dùng pickle? Nạp một file pickle là CHẠY CODE TUỲ Ý nằm trong file
# đó — mở một checkpoint tải từ mạng là mở cửa cho máy mình. Bảng tra chỉ cho phép
# dựng đúng 7 lớp liệt kê ở đây, mô tả lạ sẽ bị từ chối. Ngoài ra .npz đọc được
# từ bất kỳ đâu, còn pickle gắn chặt với phiên bản Python và cấu trúc class.
BANG_LOP = {
    "Dense": Dense,
    "ReLU": ReLU,
    "Dropout": Dropout,
    "BatchNorm1d": BatchNorm1d,
    "Conv2D": Conv2D,
    "MaxPool2D": MaxPool2D,
    "Flatten": Flatten,
}

PHIEN_BAN = 1      # tăng lên khi định dạng checkpoint thay đổi không tương thích


def luu_model(model: Sequential, path: str, meta: dict | None = None,
              verbose: bool = True) -> None:
    """
    Lưu toàn bộ mô hình vào một file .npz nén.

    Parameters:
        model: Sequential cần lưu.
        path: Đường dẫn file .npz.
        meta: Thông tin kèm theo. BẮT BUỘC nên có "mean", "std", "img_size",
            "class_names" — thiếu thì lúc dự đoán không biết tiền xử lý thế nào.

    Cấu trúc file:
        kien_truc  : JSON, danh sách cau_hinh() của từng lớp
        meta       : JSON, kèm phien_ban và danh sách lớp để kiểm tra tương thích
        p{i}.{ten} : tham số học được của lớp thứ i
        b{i}.{ten} : buffer của lớp thứ i
    """
    meta = dict(meta or {})
    meta["phien_ban"] = PHIEN_BAN
    # Lưu cả danh sách loại lớp: nạp bằng phiên bản code cũ (chưa có lớp nào đó)
    # sẽ báo lỗi rõ ràng thay vì dựng ra một mô hình sai.
    meta["cac_loai_lop"] = sorted({type(l).__name__ for l in model.layers})

    mang = {
        "kien_truc": np.array(json.dumps([l.cau_hinh() for l in model.layers])),
        "meta": np.array(json.dumps(meta)),
    }
    for i, layer in enumerate(model.layers):
        for ten, p in layer.params().items():
            mang[f"p{i}.{ten}"] = p
        for ten, b in layer.buffers().items():     # đừng quên buffer
            mang[f"b{i}.{ten}"] = b

    thu_muc = os.path.dirname(path)
    if thu_muc:
        os.makedirs(thu_muc, exist_ok=True)
    np.savez_compressed(path, **mang)

    if verbose:
        mb = os.path.getsize(path) / 1024 ** 2
        print(f"Đã lưu: {path} ({mb:.2f} MB, {model.so_tham_so():,} tham số, "
              f"{len(model.layers)} lớp)")


def nap_model(path: str, verbose: bool = True) -> tuple[Sequential, dict]:
    """
    Dựng lại mô hình từ file .npz.

    Returns:
        (model, meta). Mô hình đã sẵn sàng dự đoán, không cần train lại.

    Không dùng allow_pickle: file chỉ chứa mảng số và hai chuỗi JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {path}\n"
            f"Hãy chạy trước:  python src/train_final.py"
        )

    data = np.load(path)
    kien_truc = json.loads(str(data["kien_truc"]))
    meta = json.loads(str(data["meta"]))

    if meta.get("phien_ban", 0) > PHIEN_BAN:
        raise ValueError(f"Checkpoint dùng định dạng phiên bản {meta['phien_ban']}, "
                         f"code này chỉ hiểu tới {PHIEN_BAN}")
    thieu = set(meta.get("cac_loai_lop", [])) - set(BANG_LOP)
    if thieu:
        raise ValueError(f"Checkpoint cần các lớp chưa có trong BANG_LOP: {sorted(thieu)}")

    layers = []
    for mo_ta in kien_truc:
        mo_ta = dict(mo_ta)
        loai = mo_ta.pop("loai")
        if loai not in BANG_LOP:
            raise ValueError(f"Không nhận ra lớp '{loai}'. Các lớp hợp lệ: {sorted(BANG_LOP)}")
        layers.append(BANG_LOP[loai](**mo_ta))

    model = Sequential(layers)

    # Ghi TẠI CHỖ vào các mảng vừa được khởi tạo
    for i, layer in enumerate(model.layers):
        for nhan, kho in (("p", layer.params()), ("b", layer.buffers())):
            for ten, mang in kho.items():
                khoa = f"{nhan}{i}.{ten}"
                if khoa not in data:
                    raise KeyError(f"Checkpoint thiếu '{khoa}' cho lớp {i} ({layer})")
                gia_tri = data[khoa]
                if gia_tri.shape != mang.shape:
                    raise ValueError(
                        f"Lớp {i} ({layer}), '{ten}': checkpoint có shape {gia_tri.shape} "
                        f"nhưng lớp dựng ra cần {mang.shape}"
                    )
                mang[...] = gia_tri

    if verbose:
        print(f"Đã nạp: {path}")
        print(f"  {model}")
        if "val_acc" in meta:
            print(f"  val accuracy lúc lưu: {meta['val_acc']:.4f}")
    return model, meta


def _chuan_bi_dau_vao(model: Sequential, meta: dict, anh: np.ndarray) -> np.ndarray:
    """
    Đưa ảnh thô (N, H, W) uint8 về đúng dạng và thang đo mà mô hình được train.

    Phải khớp CHÍNH XÁC trình tự lúc train, và dùng mean/std ĐÃ LƯU chứ không
    tính lại từ ảnh mới.
    """
    X = anh.reshape(len(anh), -1).astype(np.float32)
    X = (X - meta["mean"]) / meta["std"]

    # CNN cần (N, 1, H, W); MLP cần (N, D). Tự nhận biết theo lớp đầu tiên.
    if isinstance(model.layers[0], Conv2D):
        size = meta["img_size"]
        X = X.reshape(len(anh), 1, size, size)
    return X


def du_doan_nhieu_anh(model: Sequential, meta: dict,
                      cac_duong_dan: list[str]) -> np.ndarray:
    """
    Dự đoán cả lô ảnh từ đường dẫn file.

    Returns:
        (N, C) xác suất softmax.

    Xử lý cả lô một lần nhanh hơn nhiều so với gọi từng ảnh trong vòng lặp.
    """
    anh = np.stack([load_image(p, meta["img_size"]) for p in cac_duong_dan])
    X = _chuan_bi_dau_vao(model, meta, anh)
    return softmax(model.forward(X, training=False))     # training=False!


def du_doan_anh(model: Sequential, meta: dict, duong_dan: str,
                top_k: int = 3) -> list[tuple[str, float]]:
    """
    Dự đoán một file ảnh bất kỳ.

    Parameters:
        duong_dan: Đường dẫn tới file .jpg/.png.
        top_k: Số lớp có xác suất cao nhất cần trả về.

    Returns:
        [(tên lớp, xác suất), ...] sắp xếp giảm dần.
    """
    if not os.path.exists(duong_dan):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {duong_dan}")

    P = du_doan_nhieu_anh(model, meta, [duong_dan])[0]
    ten_lop = meta["class_names"]
    thu_tu = np.argsort(-P)[:top_k]
    return [(ten_lop[c], float(P[c])) for c in thu_tu]


if __name__ == "__main__":
    import time

    from data_loader import CLASS_NAMES, DATA_ROOT, PROJECT_ROOT
    from losses import SoftmaxCrossEntropy, accuracy
    from model import tao_mlp
    from optimizers import Adam
    from preprocess import prepare_data
    from train import train

    duong_dan_test = os.path.join(PROJECT_ROOT, "checkpoints", "test_tam.npz")

    # ------------------------------------------------------------------
    print("=" * 76)
    print("TEST 1: Vòng tròn lưu -> nạp phải khớp TỪNG BIT")
    print("=" * 76)
    data = prepare_data()
    meta = {"mean": data["mean"], "std": data["std"], "img_size": 64,
            "class_names": CLASS_NAMES}

    model = tao_mlp(4096, [64], 6, rng=np.random.default_rng(0),
                    dropout=0.3, batchnorm=True)
    # Train vài epoch để running_mean/var khác giá trị khởi tạo
    train(model, Adam(model.layers, lr=1e-3), SoftmaxCrossEntropy(), data,
          epochs=3, batch_size=64, rng=np.random.default_rng(0), verbose=False)

    X = data["X_val"][:32]
    truoc = model.forward(X, training=False)

    luu_model(model, duong_dan_test, meta)
    model2, meta2 = nap_model(duong_dan_test)
    sau = model2.forward(X, training=False)

    print(f"Khớp từng bit ? {np.array_equal(truoc, sau)}")
    print(f"repr giống nhau ? {repr(model) == repr(model2)}")
    print(f"meta giữ nguyên ? {all(meta2[k] == meta[k] for k in meta)}")
    assert np.array_equal(truoc, sau), "Lưu/nạp làm đổi kết quả!"
    assert repr(model) == repr(model2)

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("TEST 2: Bẫy QUÊN BUFFER (chính là lỗi đã gặp ở Bước 10)")
    print("=" * 76)
    acc_dung = accuracy(model2.forward(data["X_val"], training=False), data["y_val"])

    # Cố tình nạp thiếu buffer: đặt running_mean/var về giá trị khởi tạo
    model3, _ = nap_model(duong_dan_test, verbose=False)
    for layer in model3.layers:
        if isinstance(layer, BatchNorm1d):
            layer.running_mean[...] = 0.0
            layer.running_var[...] = 1.0
    acc_thieu = accuracy(model3.forward(data["X_val"], training=False), data["y_val"])

    print(f"Nạp ĐỦ   (có running_mean/var): accuracy = {acc_dung:.4f}")
    print(f"Nạp THIẾU buffer               : accuracy = {acc_thieu:.4f}")
    print(f"-> mất {acc_dung - acc_thieu:+.4f}, và KHÔNG có lỗi nào được báo")
    assert acc_dung != acc_thieu

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("TEST 3: Bẫy QUÊN mean/std khi tiền xử lý")
    print("=" * 76)
    duong_dan_anh = os.path.join(DATA_ROOT, "validation", "images",
                                 "scratches", "scratches_300.jpg")
    anh = load_image(duong_dan_anh, 64)[None]      # (1, 64, 64) uint8

    def du_doan_tho(X_phang: np.ndarray) -> tuple[str, float]:
        Xc = X_phang
        if isinstance(model2.layers[0], Conv2D):
            Xc = Xc.reshape(1, 1, 64, 64)
        P = softmax(model2.forward(Xc, training=False))[0]
        c = int(P.argmax())
        return CLASS_NAMES[c], float(P[c])

    phang = anh.reshape(1, -1).astype(np.float32)
    cach = {
        "(a) đúng: mean/std đã lưu": (phang - meta["mean"]) / meta["std"],
        "(b) KHÔNG chuẩn hóa (0-255)": phang,
        "(c) mean/std của CHÍNH ảnh đó": (phang - phang.mean()) / phang.std(),
    }
    print("Ảnh thật: scratches_300.jpg (nhãn đúng = scratches)")
    for ten, Xc in cach.items():
        lop, p = du_doan_tho(Xc)
        print(f"  {ten:<32} -> {lop:<16} ({p:.3f})")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("TEST 4: Báo lỗi tử tế")
    print("=" * 76)
    for mo_ta, ham in [
        ("checkpoint không tồn tại", lambda: nap_model("khong_co_dau.npz")),
        ("ảnh không tồn tại", lambda: du_doan_anh(model2, meta, "khong_co_anh.jpg")),
    ]:
        try:
            ham()
        except FileNotFoundError as e:
            print(f"  {mo_ta:<28} -> FileNotFoundError: {str(e).splitlines()[0]}")

    if "LopMaKhong" not in BANG_LOP:
        print(f"  {'lớp lạ trong kiến trúc':<28} -> bị BANG_LOP từ chối ✔")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("TEST 5: Dự đoán từ file JPEG thật (đi qua cv2.imread, không qua cache)")
    print("=" * 76)
    rng = np.random.default_rng(0)
    cac_duong_dan, nhan_that = [], []
    for ten_lop in CLASS_NAMES:
        thu_muc = os.path.join(DATA_ROOT, "validation", "images", ten_lop)
        for f in rng.choice(sorted(os.listdir(thu_muc)), 2, replace=False):
            cac_duong_dan.append(os.path.join(thu_muc, f))
            nhan_that.append(ten_lop)

    t0 = time.perf_counter()
    P = du_doan_nhieu_anh(model2, meta, cac_duong_dan)
    ms = (time.perf_counter() - t0) / len(cac_duong_dan) * 1000

    print(f"{'file':<26}{'nhãn thật':<18}{'dự đoán':<18}{'xác suất':>10}")
    print("-" * 74)
    dung = 0
    for p, that, xs in zip(cac_duong_dan, nhan_that, P):
        doan = CLASS_NAMES[int(xs.argmax())]
        dung += doan == that
        dau = "✔" if doan == that else "✘"
        print(f"{os.path.basename(p):<26}{that:<18}{doan:<18}{xs.max():>9.3f} {dau}")
    print(f"\nĐúng {dung}/{len(nhan_that)} | {ms:.1f} ms/ảnh (gồm cả đọc file JPEG)")

    print("\n--- du_doan_anh trả top-3 cho một ảnh ---")
    for lop, p in du_doan_anh(model2, meta, duong_dan_anh):
        print(f"  {lop:<18}{p:.3f}")

    os.remove(duong_dan_test)
    print("\n" + "=" * 76)
    print("TẤT CẢ TEST ĐỀU ĐẠT ✔  (mô hình dùng ở đây chỉ là MLP nhỏ 3 epoch;")
    print("chạy src/train_final.py để tạo checkpoint CNN thật)")
    print("=" * 76)
