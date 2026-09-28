"""
Bước 10 — So sánh các kỹ thuật chống overfit trên NEU-DET.

Mốc từ Bước 8/9 (không chống gì):
    train acc 0.943 | val acc 0.481 | chênh lệch +0.42 | val loss 2.07 (cao hơn ln6!)

Mỗi cấu hình dùng CÙNG seed 42, nên cùng bộ trọng số ban đầu và cùng thứ tự xáo
dữ liệu. Khác biệt duy nhất là kỹ thuật chống overfit.
"""

import os

import numpy as np

from data_loader import CLASS_NAMES, PROJECT_ROOT, load_cache
from evaluate import danh_gia_day_du, ve_confusion_matrix, xem_anh_sai
from losses import SoftmaxCrossEntropy
from model import tao_mlp
from optimizers import Adam
from preprocess import prepare_data
from train import train, ve_duong_cong

# (tên, hidden, dropout, batchnorm, weight_decay, augment)
CAC_CAU_HINH = [
    ("1. Cơ sở (Bước 8)",          [128], 0.0, False, 0.0,  False),
    ("2. + L2 1e-4",               [128], 0.0, False, 1e-4, False),
    ("3. + L2 1e-3",               [128], 0.0, False, 1e-3, False),
    ("4. + Dropout 0.5",           [128], 0.5, False, 0.0,  False),
    ("5. + Dropout 0.2",           [128], 0.2, False, 0.0,  False),
    ("6. + BatchNorm",             [128], 0.0, True,  0.0,  False),
    ("7. + Augment (lật)",         [128], 0.0, False, 0.0,  True),
    ("8. BN + Drop0.5 + L2 + Aug", [128], 0.5, True,  1e-4, True),
    ("9. Như 8, hidden [256,128]", [256, 128], 0.5, True, 1e-4, True),
]


def chay_mot_cau_hinh(data: dict, hidden: list[int], dropout: float, batchnorm: bool,
                      weight_decay: float, augment: bool, epochs: int = 60,
                      batch_size: int = 64, lr: float = 1e-3, seed: int = 42,
                      verbose: bool = False) -> tuple:
    """
    Dựng mạng theo cấu hình, train, trả về (model, history).

    Không dùng early stopping để dừng (patience=None) nhưng VẪN khôi phục bộ
    trọng số của epoch tốt nhất, để con số báo cáo là thứ ta thực sự dùng được.
    """
    model = tao_mlp(data["X_train"].shape[1], hidden, len(CLASS_NAMES),
                    rng=np.random.default_rng(seed),
                    dropout=dropout, batchnorm=batchnorm)
    opt = Adam(model.layers, lr=lr, weight_decay=weight_decay)   # decoupled -> AdamW
    history = train(model, opt, SoftmaxCrossEntropy(), data,
                    epochs=epochs, batch_size=batch_size,
                    rng=np.random.default_rng(seed), verbose=verbose,
                    khoi_phuc_tot_nhat=True, augment=augment)
    return model, history


def so_sanh_chong_overfit(data: dict, epochs: int = 60) -> dict:
    """Chạy toàn bộ CAC_CAU_HINH và in bảng so sánh. Trả về dict tên -> history."""
    print(f"{'cấu hình':<30}{'tham số':>11}{'train':>8}{'val':>8}"
          f"{'val tốt nhất':>14}{'chênh lệch':>12}{'epoch tốt':>11}{'val loss':>10}")
    print("-" * 104)

    ket_qua = {}
    for ten, hidden, dropout, bn, wd, aug in CAC_CAU_HINH:
        model, h = chay_mot_cau_hinh(data, hidden, dropout, bn, wd, aug, epochs=epochs)
        ket_qua[ten] = (model, h)

        if h["no_o_epoch"] is not None:
            print(f"{ten:<30}{model.so_tham_so():>11,}{'NỔ ở epoch ' + str(h['no_o_epoch']):>63}")
            continue

        # Sau khôi phục, chỉ số "cuối" lấy tại epoch tốt nhất
        i = h["epoch_tot_nhat"] - 1
        print(f"{ten:<30}{model.so_tham_so():>11,}{h['train_acc'][i]:>8.4f}"
              f"{h['val_acc'][-1]:>8.4f}{h['val_acc_tot_nhat']:>14.4f}"
              f"{h['train_acc'][i] - h['val_acc_tot_nhat']:>+12.4f}"
              f"{h['epoch_tot_nhat']:>11}{min(h['val_loss']):>10.4f}")

    return ket_qua


if __name__ == "__main__":
    data = prepare_data()

    print("=" * 104)
    print("SO SÁNH CÁC KỸ THUẬT CHỐNG OVERFIT — MLP, Adam lr=1e-3, 60 epoch, seed 42")
    print("(cột 'train' và 'chênh lệch' lấy tại epoch có val tốt nhất)")
    print("=" * 104)
    ket_qua = so_sanh_chong_overfit(data, epochs=60)

    # ------------------------------------------------------------------
    # Chọn cấu hình có val accuracy tốt nhất rồi mổ xẻ kỹ
    ten_tot_nhat = max(ket_qua, key=lambda k: ket_qua[k][1]["val_acc_tot_nhat"])
    model, history = ket_qua[ten_tot_nhat]

    print("\n" + "=" * 104)
    print(f"CẤU HÌNH TỐT NHẤT: {ten_tot_nhat}")
    print("=" * 104)
    print(model)

    criterion = SoftmaxCrossEntropy()
    danh_gia_day_du(model, criterion, data["X_train"], data["Y_train"],
                    data["y_train"], CLASS_NAMES, ten_tap="train")
    kq = danh_gia_day_du(model, criterion, data["X_val"], data["Y_val"],
                         data["y_val"], CLASS_NAMES, ten_tap="val")

    thu_muc = os.path.join(PROJECT_ROOT, "outputs")
    ve_duong_cong(history, save_path=os.path.join(thu_muc, "training_curve_chong_overfit.png"),
                  show=False, tieu_de=f"NEU-DET — {ten_tot_nhat}")
    ve_confusion_matrix(kq["cm"], CLASS_NAMES, show=False,
                        save_path=os.path.join(thu_muc, "confusion_matrix_chong_overfit.png"),
                        tieu_de=f"val — {ten_tot_nhat} — accuracy {kq['accuracy']:.3f}")
    _, _, X_val_raw, _ = load_cache()
    xem_anh_sai(model, data["X_val"], X_val_raw, data["y_val"], CLASS_NAMES, n=12,
                show=False, save_path=os.path.join(thu_muc, "anh_sai_chong_overfit.png"))

    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("SO VỚI MỐC BƯỚC 8/9 (không chống overfit)")
    print("=" * 104)
    co_so = ket_qua["1. Cơ sở (Bước 8)"][1]
    print(f"{'':<22}{'cơ sở':>12}{'tốt nhất':>12}{'thay đổi':>12}")
    print("-" * 58)
    for nhan, khoa in [("val accuracy", "val_acc_tot_nhat")]:
        a, b = co_so[khoa], history[khoa]
        print(f"{nhan:<22}{a:>12.4f}{b:>12.4f}{b - a:>+12.4f}")
    a, b = min(co_so["val_loss"]), min(history["val_loss"])
    print(f"{'val loss thấp nhất':<22}{a:>12.4f}{b:>12.4f}{b - a:>+12.4f}")
    a = co_so["train_acc"][co_so["epoch_tot_nhat"] - 1] - co_so["val_acc_tot_nhat"]
    b = history["train_acc"][history["epoch_tot_nhat"] - 1] - history["val_acc_tot_nhat"]
    print(f"{'chênh lệch train-val':<22}{a:>+12.4f}{b:>+12.4f}{b - a:>+12.4f}")
