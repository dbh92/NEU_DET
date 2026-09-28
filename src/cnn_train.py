"""
Bước 11 — Huấn luyện CNN trên NEU-DET và so với MLP.

Mốc cần vượt (Bước 10, MLP [256,128] + BN + Dropout + AdamW + lật):
    val accuracy 0.611 | val loss 0.876 | chênh lệch train-val +0.069
    NHƯNG rolled-in_scale vẫn recall 0.000 — đó mới là thứ cần chữa.
"""

import os
import time

import numpy as np

from conv import Conv2D
from data_loader import CLASS_NAMES, PROJECT_ROOT, load_cache
from evaluate import danh_gia_day_du, ve_confusion_matrix
from losses import SoftmaxCrossEntropy
from model import tao_cnn, tao_mlp
from optimizers import Adam
from preprocess import prepare_data
from train import train, ve_duong_cong


def du_lieu_dang_anh(data: dict, img_size: int = 64) -> dict:
    """
    Đổi X từ (N, 4096) sang (N, 1, 64, 64) cho CNN. Nhãn giữ nguyên.

    reshape KHÔNG sao chép dữ liệu, chỉ đổi cách diễn giải cùng vùng nhớ, nên
    thao tác này gần như miễn phí.
    """
    moi = dict(data)
    for ten in ("X_train", "X_val"):
        moi[ten] = data[ten].reshape(-1, 1, img_size, img_size)
    return moi


def ve_bo_loc(conv: Conv2D, save_path: str | None = None, show: bool = True) -> None:
    """
    Vẽ các bộ lọc của lớp Conv đầu tiên — xem mạng đã học "nhìn" cái gì.

    Bộ lọc 3x3 quá nhỏ để đẹp mắt, nhưng vẫn thấy được mẫu hình: dải sáng/tối
    theo chiều dọc = bộ dò cạnh dọc, theo chiều ngang = dò cạnh ngang, v.v.
    """
    import matplotlib.pyplot as plt

    W = conv.W                                   # (C_out, C_in, KH, KW)
    n = W.shape[0]
    cot = min(n, 8)
    hang = int(np.ceil(n / cot))

    fig, axes = plt.subplots(hang, cot, figsize=(1.4 * cot, 1.6 * hang))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k in range(n):
        # Cùng thang màu cho mọi bộ lọc để so sánh được độ mạnh giữa chúng
        axes[k].imshow(W[k, 0], cmap="gray", vmin=W.min(), vmax=W.max())
        axes[k].set_title(f"#{k}", fontsize=8)

    fig.suptitle(f"{n} bộ lọc {W.shape[2]}x{W.shape[3]} của lớp Conv đầu tiên")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"Đã lưu hình: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def chay_cnn(data_anh: dict, kenh=(8, 16), hidden: int = 64, dropout: float = 0.5,
             batchnorm: bool = False, augment: bool = True, epochs: int = 40,
             lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 64,
             seed: int = 42, verbose: bool = False) -> tuple:
    """Dựng CNN theo cấu hình, train, trả về (model, history, thời gian)."""
    model = tao_cnn(kenh=kenh, hidden=hidden, rng=np.random.default_rng(seed),
                    dropout=dropout, batchnorm=batchnorm,
                    num_classes=len(CLASS_NAMES))
    opt = Adam(model.layers, lr=lr, weight_decay=weight_decay)

    t0 = time.perf_counter()
    h = train(model, opt, SoftmaxCrossEntropy(), data_anh,
              epochs=epochs, batch_size=batch_size,
              rng=np.random.default_rng(seed), verbose=verbose,
              khoi_phuc_tot_nhat=True, augment=augment)
    return model, h, time.perf_counter() - t0


if __name__ == "__main__":
    EPOCHS = 40
    data = prepare_data()
    data_anh = du_lieu_dang_anh(data)
    print(f"Dữ liệu cho CNN: X_train {data_anh['X_train'].shape} | "
          f"X_val {data_anh['X_val'].shape}\n")

    ket_qua = {}

    # ------------------------------------------------------------------
    print("=" * 100)
    print(f"SO SÁNH CÁC CẤU HÌNH CNN ({EPOCHS} epoch, Adam lr=1e-3, wd=1e-4, batch 64, seed 42)")
    print("=" * 100)
    print(f"{'cấu hình':<34}{'tham số':>11}{'train':>8}{'val tốt nhất':>14}"
          f"{'chênh lệch':>12}{'epoch tốt':>11}{'val loss':>10}{'thời gian':>11}")
    print("-" * 100)

    cau_hinh = [
        ("CNN (8,16), không augment", dict(kenh=(8, 16), augment=False)),
        ("CNN (8,16) + augment",      dict(kenh=(8, 16), augment=True)),
        ("CNN (16,32) + augment",     dict(kenh=(16, 32), augment=True)),
        ("CNN (16,32) + aug + BN",    dict(kenh=(16, 32), augment=True, batchnorm=True)),
    ]
    for ten, kw in cau_hinh:
        model, h, giay = chay_cnn(data_anh, epochs=EPOCHS, **kw)
        ket_qua[ten] = (model, h)
        i = h["epoch_tot_nhat"] - 1
        print(f"{ten:<34}{model.so_tham_so():>11,}{h['train_acc'][i]:>8.4f}"
              f"{h['val_acc_tot_nhat']:>14.4f}"
              f"{h['train_acc'][i] - h['val_acc_tot_nhat']:>+12.4f}"
              f"{h['epoch_tot_nhat']:>11}{min(h['val_loss']):>10.4f}{giay:>10.0f}s")

    # Mốc so sánh: MLP tốt nhất của Bước 10, train lại đúng số epoch
    print("-" * 100)
    mlp = tao_mlp(data["X_train"].shape[1], [256, 128], len(CLASS_NAMES),
                  rng=np.random.default_rng(42), dropout=0.5, batchnorm=True)
    t0 = time.perf_counter()
    h_mlp = train(mlp, Adam(mlp.layers, lr=1e-3, weight_decay=1e-4),
                  SoftmaxCrossEntropy(), data, epochs=EPOCHS, batch_size=64,
                  rng=np.random.default_rng(42), verbose=False, augment=True)
    giay = time.perf_counter() - t0
    i = h_mlp["epoch_tot_nhat"] - 1
    print(f"{'MLP [256,128] (mốc Bước 10)':<34}{mlp.so_tham_so():>11,}"
          f"{h_mlp['train_acc'][i]:>8.4f}{h_mlp['val_acc_tot_nhat']:>14.4f}"
          f"{h_mlp['train_acc'][i] - h_mlp['val_acc_tot_nhat']:>+12.4f}"
          f"{h_mlp['epoch_tot_nhat']:>11}{min(h_mlp['val_loss']):>10.4f}{giay:>10.0f}s")

    # ------------------------------------------------------------------
    ten_tot = max(ket_qua, key=lambda k: ket_qua[k][1]["val_acc_tot_nhat"])
    model, history = ket_qua[ten_tot]

    print("\n" + "=" * 100)
    print(f"CNN TỐT NHẤT: {ten_tot}")
    print("=" * 100)
    print(model)
    conv1 = model.layers[0]
    print(f"\nPhần tích chập chỉ tốn "
          f"{sum(l.W.size + l.b.size for l in model.layers if isinstance(l, Conv2D)):,} tham số")

    criterion = SoftmaxCrossEntropy()
    kq = danh_gia_day_du(model, criterion, data_anh["X_val"], data_anh["Y_val"],
                         data_anh["y_val"], CLASS_NAMES, ten_tap="val")

    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("CÂU HỎI CHÍNH CỦA BƯỚC 11: CNN có cứu được rolled-in_scale không?")
    print("=" * 100)
    print(f"{'lớp':<18}{'recall MLP (B10)':>18}{'recall CNN':>14}{'thay đổi':>12}")
    print("-" * 62)
    kq_mlp = danh_gia_day_du(mlp, criterion, data["X_val"], data["Y_val"],
                             data["y_val"], CLASS_NAMES, ten_tap="val (MLP mốc)")
    print()
    for c, ten_lop in enumerate(CLASS_NAMES):
        a, b = kq_mlp["recall"][c], kq["recall"][c]
        print(f"{ten_lop:<18}{a:>18.3f}{b:>14.3f}{b - a:>+12.3f}")

    # ------------------------------------------------------------------
    thu_muc = os.path.join(PROJECT_ROOT, "outputs")
    ve_duong_cong(history, save_path=os.path.join(thu_muc, "training_curve_cnn.png"),
                  show=False, tieu_de=f"NEU-DET — {ten_tot}")
    ve_confusion_matrix(kq["cm"], CLASS_NAMES, show=False,
                        save_path=os.path.join(thu_muc, "confusion_matrix_cnn.png"),
                        tieu_de=f"val — {ten_tot} — accuracy {kq['accuracy']:.3f}")
    ve_bo_loc(conv1, save_path=os.path.join(thu_muc, "bo_loc_conv1.png"), show=False)
