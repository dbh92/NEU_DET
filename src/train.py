"""
Vòng lặp huấn luyện: train nhiều epoch trên 1440 ảnh, đánh giá trên 360 ảnh val,
ghi lại lịch sử và vẽ đường cong.

Đây là bước đầu tiên nhìn thấy OVERFITTING bằng mắt: mạng có 525,190 tham số
mà chỉ có 1440 ảnh để học, nên nó đủ sức HỌC THUỘC tập train thay vì học quy luật.
"""

import os
import time

import numpy as np

from data_loader import CLASS_NAMES, PROJECT_ROOT
from losses import SoftmaxCrossEntropy
from model import Sequential, tao_mlp
from optimizers import Adam, Momentum, Optimizer, SGD
from preprocess import iterate_minibatches, prepare_data


def danh_gia(model: Sequential, criterion: SoftmaxCrossEntropy,
             X: np.ndarray, Y: np.ndarray, y: np.ndarray,
             batch_size: int = 256) -> tuple[float, float]:
    """
    Tính loss và accuracy trên toàn bộ một tập dữ liệu, KHÔNG cập nhật tham số.

    Parameters:
        X: (N, D) đã chuẩn hóa. Y: (N, C) one-hot. y: (N,) nhãn số.
        batch_size: Chia nhỏ để không nuốt hết RAM khi N lớn.

    Returns:
        (loss, accuracy) trung bình trên đúng N mẫu.

    Cộng dồn CÓ TRỌNG SỐ theo số mẫu: batch cuối thường ngắn hơn, nếu lấy trung
    bình của các trung bình thì batch cuối bị tính nặng ngang batch đầy -> sai.
    """
    tong_loss, tong_dung, n = 0.0, 0, 0

    # shuffle=False: tập đánh giá không cần xáo, và để kết quả tái lập được
    for Xb, Yb in iterate_minibatches(X, Y, batch_size, shuffle=False):
        Z = model.forward(Xb, training=False)       # chế độ ĐÁNH GIÁ
        tong_loss += criterion.forward(Z, Yb) * len(Xb)
        tong_dung += int((Z.argmax(axis=1) == y[n:n + len(Xb)]).sum())
        n += len(Xb)

    return tong_loss / n, tong_dung / n


def lat_ngau_nhien(Xb: np.ndarray, rng: np.random.Generator, size: int = 64) -> np.ndarray:
    """
    Tăng cường dữ liệu: lật ngang và/hoặc dọc từng ảnh một cách ngẫu nhiên.

    Parameters:
        Xb: (B, size*size) đã duỗi phẳng cho MLP, HOẶC (B, C, H, W) cho CNN.
        rng: Bộ sinh ngẫu nhiên.
        size: Cạnh ảnh vuông (chỉ dùng khi Xb ở dạng phẳng).

    Returns:
        Cùng shape và dtype với Xb.

    Ảnh lỗi bề mặt thép KHÔNG có chiều "đúng": một vết xước lật ngược vẫn là vết
    xước. Vậy mỗi ảnh có 4 biến thể hợp lệ (gốc, lật ngang, lật dọc, lật cả hai)
    -> nhân dữ liệu lên 4 lần mà không tốn thêm ảnh nào.

    Mỗi ảnh được quyết định ĐỘC LẬP, và quyết định lại ở mỗi epoch, nên mạng gần
    như không bao giờ thấy đúng cùng một batch hai lần.
    """
    B = Xb.shape[0]
    # Đưa cả hai dạng về (B, C, H, W) để xử lý chung
    anh = Xb.reshape(B, 1, size, size) if Xb.ndim == 2 else Xb

    ngang = rng.random(B) < 0.5
    doc = rng.random(B) < 0.5
    # [..., ::-1] lật theo chiều cuối (rộng); [..., ::-1, :] lật chiều cao
    anh = np.where(ngang[:, None, None, None], anh[..., ::-1], anh)
    anh = np.where(doc[:, None, None, None], anh[..., ::-1, :], anh)

    return anh.reshape(Xb.shape).astype(Xb.dtype)


def train(model: Sequential, optimizer: Optimizer, criterion: SoftmaxCrossEntropy,
          data: dict, epochs: int = 30, batch_size: int = 64,
          rng: np.random.Generator | None = None, verbose: bool = True,
          patience: int | None = None, khoi_phuc_tot_nhat: bool = True,
          augment: bool = False) -> dict:
    """
    Huấn luyện mô hình và trả về lịch sử để vẽ đường cong.

    Parameters:
        model: Sequential đã dựng sẵn.
        optimizer: Đã được gắn vào model.layers từ trước.
        criterion: SoftmaxCrossEntropy.
        data: dict từ prepare_data().
        epochs: Số lần duyệt hết tập train.
        batch_size: 1440 / 64 = 22.5 -> 23 batch, batch cuối có 32 ảnh.
        rng: Bộ sinh ngẫu nhiên dùng để xáo dữ liệu mỗi epoch.
        verbose: In một dòng mỗi epoch.

    Returns:
        dict gồm 4 list cùng độ dài epochs:
        "train_loss", "train_acc", "val_loss", "val_acc".
    """
    if rng is None:
        rng = np.random.default_rng()

    X_train, Y_train = data["X_train"], data["Y_train"]
    X_val, Y_val, y_val = data["X_val"], data["Y_val"], data["y_val"]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "no_o_epoch": None, "epoch_tot_nhat": None, "val_acc_tot_nhat": -1.0,
               "dung_som_o_epoch": None}
    bat_dau = time.perf_counter()

    # Early stopping: nhớ bộ trọng số tại epoch có val accuracy cao nhất
    tham_so_tot_nhat: list[np.ndarray] | None = None
    buffer_tot_nhat: list[np.ndarray] | None = None
    so_epoch_khong_tien_bo = 0

    if verbose:
        print(model)
        print(f"{optimizer} | batch_size = {batch_size} | {epochs} epoch | "
              f"{len(X_train)} ảnh train, {len(X_val)} ảnh val\n")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tong_loss, tong_dung, n = 0.0, 0, 0

        # shuffle=True: mỗi epoch một thứ tự khác nhau
        for Xb, Yb in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True, rng=rng):
            if augment:
                Xb = lat_ngau_nhien(Xb, rng)    # chỉ tăng cường lúc train

            Z = model.forward(Xb, training=True)
            loss = criterion.forward(Z, Yb)

            # Bắt mạng NỔ ngay lúc xảy ra. Nếu cứ chạy tiếp, nan sẽ lan ra toàn bộ
            # W và mọi epoch sau đều vô nghĩa, mà không có lỗi nào được báo.
            if not np.isfinite(loss):
                history["no_o_epoch"] = epoch
                break

            optimizer.zero_grad()               # chưa cần bây giờ, Bước 10 sẽ cần
            model.backward(criterion.backward())
            optimizer.step()

            tong_loss += loss * len(Xb)
            tong_dung += int((Z.argmax(axis=1) == Yb.argmax(axis=1)).sum())
            n += len(Xb)

        # train_acc tính TRONG LÚC ĐI: mỗi batch được chấm bằng bộ trọng số tại
        # thời điểm đó, mà trọng số đổi liên tục -> luôn thấp hơn một chút so với
        # chấm lại cả tập sau khi epoch kết thúc. Mọi framework đều làm vậy vì rẻ hơn.
        if history["no_o_epoch"] is not None:
            if verbose:
                print(f"epoch {epoch:3d}/{epochs} | loss = nan/inf -> MẠNG NỔ, dừng sớm. "
                      f"Learning rate quá lớn.")
            break

        train_loss, train_acc = tong_loss / n, tong_dung / n
        val_loss, val_acc = danh_gia(model, criterion, X_val, Y_val, y_val)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # --- Early stopping: theo dõi và giữ lại bộ trọng số tốt nhất ---
        pha_ky_luc = val_acc > history["val_acc_tot_nhat"]
        if pha_ky_luc:
            history["val_acc_tot_nhat"] = val_acc
            history["epoch_tot_nhat"] = epoch
            # .copy() là bắt buộc: params() trả về THAM CHIẾU, các epoch sau sẽ
            # ghi đè lên chính những mảng này.
            tham_so_tot_nhat = [p.copy() for p in model.params()]
            # PHẢI chụp cả buffer: running_mean/var của BatchNorm không phải tham số
            # học được, nhưng chúng quyết định kết quả lúc đánh giá. Chỉ khôi phục
            # W mà bỏ quên chúng thì mô hình chạy với trọng số của epoch này và
            # thống kê chuẩn hóa của epoch khác -> accuracy tụt mà không rõ vì sao.
            buffer_tot_nhat = [b.copy() for b in model.buffers()]
            so_epoch_khong_tien_bo = 0
        else:
            so_epoch_khong_tien_bo += 1

        if verbose:
            print(f"epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f}"
                  f" | val loss {val_loss:.4f} acc {val_acc:.4f}"
                  f" | {time.perf_counter() - t0:.2f}s{'  <- kỷ lục' if pha_ky_luc else ''}")

        if patience is not None and so_epoch_khong_tien_bo >= patience:
            history["dung_som_o_epoch"] = epoch
            if verbose:
                print(f"Dừng sớm: {patience} epoch liên tiếp không cải thiện val accuracy.")
            break

    # Khôi phục bộ trọng số tốt nhất: ghi TẠI CHỖ để model và optimizer vẫn
    # trỏ vào đúng những mảng cũ.
    if khoi_phuc_tot_nhat and tham_so_tot_nhat is not None:
        for p, p_tot in zip(model.params(), tham_so_tot_nhat):
            p[...] = p_tot
        for b, b_tot in zip(model.buffers(), buffer_tot_nhat):
            b[...] = b_tot
        if verbose:
            print(f"Đã khôi phục trọng số của epoch {history['epoch_tot_nhat']} "
                  f"(val acc {history['val_acc_tot_nhat']:.4f})")

    if verbose and history["val_acc"]:
        tot_nhat = int(np.argmax(history["val_acc"]))
        print(f"\nTổng thời gian: {time.perf_counter() - bat_dau:.1f}s")
        print(f"Val accuracy cao nhất: {history['val_acc'][tot_nhat]:.4f} tại epoch {tot_nhat + 1}")
        print(f"Chênh lệch cuối cùng train - val: "
              f"{history['train_acc'][-1] - history['val_acc'][-1]:+.4f}  <- càng lớn càng overfit")

    return history


def ve_duong_cong(history: dict, save_path: str | None = None, show: bool = True,
                  tieu_de: str = "NEU-DET — MLP thuần NumPy") -> None:
    """
    Vẽ loss và accuracy theo epoch cho cả train và val.

    Parameters:
        history: dict trả về từ train().
        save_path: Đường dẫn file .png để lưu, None thì không lưu.
        show: Có mở cửa sổ hiển thị hay không.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.axhline(np.log(len(CLASS_NAMES)), ls=":", c="gray", label="ln 6 (đoán mò)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.axhline(1 / len(CLASS_NAMES), ls=":", c="gray", label="1/6 (đoán mò)")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_title("Accuracy")

    for ax in (ax_loss, ax_acc):
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(tieu_de)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"Đã lưu hình: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def so_sanh(data: dict, cac_cau_hinh: list[tuple], epochs: int = 20,
            batch_size: int = 64, seed: int = 42) -> None:
    """
    Chạy nhiều cấu hình và in bảng so sánh.

    Parameters:
        cac_cau_hinh: list các tuple (tên, hidden, hàm tạo optimizer).
            Ví dụ: ("Adam lr=1e-3", [128], lambda layers: Adam(layers, lr=1e-3))
        epochs, batch_size, seed: Giữ giống nhau để so sánh công bằng.
    """
    print(f"{'cấu hình':<26}{'tham số':>12}{'train acc':>12}{'val acc':>10}"
          f"{'val acc tốt nhất':>18}{'chênh lệch':>12}")
    print("-" * 92)

    for ten, hidden, tao_opt in cac_cau_hinh:
        # Cùng seed -> cùng bộ trọng số ban đầu, cùng thứ tự xáo dữ liệu
        model = tao_mlp(data["X_train"].shape[1], hidden, len(CLASS_NAMES),
                        rng=np.random.default_rng(seed))
        h = train(model, tao_opt(model.layers), SoftmaxCrossEntropy(), data,
                  epochs=epochs, batch_size=batch_size,
                  rng=np.random.default_rng(seed), verbose=False)

        if h["no_o_epoch"] is not None:
            print(f"{ten:<26}{model.so_tham_so():>12,}"
                  f"{'NỔ ở epoch ' + str(h['no_o_epoch']):>52}")
            continue

        print(f"{ten:<26}{model.so_tham_so():>12,}{h['train_acc'][-1]:>12.4f}"
              f"{h['val_acc'][-1]:>10.4f}{max(h['val_acc']):>18.4f}"
              f"{h['train_acc'][-1] - h['val_acc'][-1]:>+12.4f}")


if __name__ == "__main__":
    data = prepare_data()

    # ==================================================================
    print("=" * 92)
    print("THÍ NGHIỆM CHÍNH: MLP 4096 -> 128 -> 6, Adam(lr=1e-3), 30 epoch")
    print("=" * 92)
    model = tao_mlp(data["X_train"].shape[1], [128], len(CLASS_NAMES),
                    rng=np.random.default_rng(42))
    history = train(model, Adam(model.layers, lr=1e-3), SoftmaxCrossEntropy(), data,
                    epochs=30, batch_size=64, rng=np.random.default_rng(42))
    ve_duong_cong(history, save_path=os.path.join(PROJECT_ROOT, "outputs", "training_curve.png"))

    # ==================================================================
    print("\n" + "=" * 92)
    print("SO SÁNH 1: learning rate (Adam, hidden=[128], 20 epoch)")
    print("=" * 92)
    so_sanh(data, [
        ("Adam lr=1e-2", [128], lambda ls: Adam(ls, lr=1e-2)),
        ("Adam lr=1e-3", [128], lambda ls: Adam(ls, lr=1e-3)),
        ("Adam lr=1e-4", [128], lambda ls: Adam(ls, lr=1e-4)),
        # lr=0.1 học thuộc 64 ảnh rất tốt ở Bước 7, nhưng trên cả 1440 ảnh thì NỔ:
        # gradient của một batch "khó" đủ lớn để hất trọng số ra khỏi vùng an toàn.
        ("SGD lr=0.1", [128], lambda ls: SGD(ls, lr=0.1)),
        ("SGD lr=0.01", [128], lambda ls: SGD(ls, lr=0.01)),
        ("Momentum lr=0.01", [128], lambda ls: Momentum(ls, lr=0.01)),
        ("Momentum lr=0.001", [128], lambda ls: Momentum(ls, lr=0.001)),
    ])

    # ==================================================================
    print("\n" + "=" * 92)
    print("SO SÁNH 2: kiến trúc (Adam lr=1e-3, 20 epoch)")
    print("=" * 92)
    so_sanh(data, [
        ("tuyến tính (không ẩn)", [], lambda ls: Adam(ls, lr=1e-3)),
        ("1 lớp ẩn [64]", [64], lambda ls: Adam(ls, lr=1e-3)),
        ("1 lớp ẩn [128]", [128], lambda ls: Adam(ls, lr=1e-3)),
        ("1 lớp ẩn [512]", [512], lambda ls: Adam(ls, lr=1e-3)),
        ("2 lớp ẩn [256, 128]", [256, 128], lambda ls: Adam(ls, lr=1e-3)),
    ])
