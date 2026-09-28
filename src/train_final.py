"""
Huấn luyện mô hình cuối cùng và lưu thành checkpoint dùng được mãi.

Cấu hình tốt nhất tìm được ở Bước 11: CNN (16,32) + augment + BatchNorm,
val accuracy 0.9417.

Chạy:  python -u src/train_final.py
Kết quả: checkpoints/cnn_neu_det.npz  (đã nằm trong .gitignore)
"""

import os
from datetime import date

import numpy as np

from checkpoint import luu_model, nap_model
from cnn_train import du_lieu_dang_anh
from data_loader import CLASS_NAMES, IMG_SIZE, PROJECT_ROOT
from evaluate import danh_gia_day_du
from losses import SoftmaxCrossEntropy, accuracy
from model import tao_cnn
from optimizers import Adam
from preprocess import prepare_data
from train import train

DUONG_DAN_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "cnn_neu_det.npz")


def train_va_luu(epochs: int = 40, seed: int = 42,
                 path: str = DUONG_DAN_CHECKPOINT) -> tuple:
    """
    Train CNN tốt nhất rồi lưu kèm đầy đủ thông tin tiền xử lý.

    Returns:
        (model, meta, history)
    """
    data = prepare_data()
    data_anh = du_lieu_dang_anh(data, IMG_SIZE)

    model = tao_cnn(kenh=(16, 32), hidden=64, img_size=IMG_SIZE,
                    num_classes=len(CLASS_NAMES), rng=np.random.default_rng(seed),
                    dropout=0.5, batchnorm=True)
    print(model, "\n")

    history = train(model, Adam(model.layers, lr=1e-3, weight_decay=1e-4),
                    SoftmaxCrossEntropy(), data_anh,
                    epochs=epochs, batch_size=64, rng=np.random.default_rng(seed),
                    verbose=True, khoi_phuc_tot_nhat=True, augment=True)

    # meta: thiếu mean/std/img_size thì mô hình vô dụng với ảnh mới
    meta = {
        "mean": float(data["mean"]),
        "std": float(data["std"]),
        "img_size": IMG_SIZE,
        "class_names": list(CLASS_NAMES),
        "val_acc": float(history["val_acc_tot_nhat"]),
        "epoch_tot_nhat": history["epoch_tot_nhat"],
        "ngay_tao": str(date.today()),
        "mo_ta": "CNN (16,32) + augment lật + BatchNorm, Adam lr=1e-3 wd=1e-4",
    }
    luu_model(model, path, meta)
    return model, meta, history


if __name__ == "__main__":
    print("=" * 76)
    print("HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG")
    print("=" * 76)
    model, meta, history = train_va_luu()

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("KIỂM CHỨNG: nạp lại từ file phải cho ĐÚNG kết quả như lúc train")
    print("=" * 76)
    data = prepare_data()
    data_anh = du_lieu_dang_anh(data, meta["img_size"])

    model_nap, meta_nap = nap_model(DUONG_DAN_CHECKPOINT)
    acc_goc = accuracy(model.forward(data_anh["X_val"], training=False), data["y_val"])
    acc_nap = accuracy(model_nap.forward(data_anh["X_val"], training=False), data["y_val"])

    print(f"\naccuracy mô hình trong RAM : {acc_goc:.4f}")
    print(f"accuracy mô hình nạp từ file: {acc_nap:.4f}")
    print(f"val_acc ghi trong meta      : {meta['val_acc']:.4f}")
    assert acc_goc == acc_nap, "Nạp lại cho kết quả khác — lưu thiếu thứ gì đó!"
    assert abs(acc_nap - meta["val_acc"]) < 1e-9
    print("Khớp tuyệt đối ✔")

    danh_gia_day_du(model_nap, SoftmaxCrossEntropy(), data_anh["X_val"],
                    data_anh["Y_val"], data["y_val"], CLASS_NAMES, ten_tap="val")

    print(f"\nCheckpoint sẵn sàng: {DUONG_DAN_CHECKPOINT}")
    print("Dùng thử:  python src/predict.py <đường dẫn ảnh .jpg>")
