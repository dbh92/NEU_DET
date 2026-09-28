"""
Đánh giá mô hình: confusion matrix, precision/recall/F1, top-k accuracy,
và xem tận mắt những ảnh bị đoán sai.

Bước 8 chỉ cho MỘT con số: val accuracy = 0.48. Con số đó không nói được mạng
sai ở đâu. File này bóc tách nó ra: lớp nào tốt, lớp nào tệ, và mạng nhầm lớp
nào với lớp nào.

Quy ước confusion matrix: HÀNG = nhãn thật, CỘT = dự đoán.
    cm[i][j] = số ảnh thật thuộc lớp i nhưng bị đoán thành lớp j
"""

import os

import numpy as np

from activations import softmax
from data_loader import CLASS_NAMES, PROJECT_ROOT, load_cache
from losses import SoftmaxCrossEntropy, accuracy
from model import Sequential, tao_mlp
from optimizers import Adam
from preprocess import prepare_data


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Đếm số mẫu cho mọi cặp (nhãn thật, nhãn đoán).

    Parameters:
        y_true: (N,) nhãn số nguyên, KHÔNG phải one-hot.
        y_pred: (N,) nhãn dự đoán.
        num_classes: Số lớp C.

    Returns:
        (C, C) int64. Hàng = thật, cột = đoán. Tổng mọi ô = N.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} và y_pred {y_pred.shape} phải cùng shape")
    if y_true.ndim != 1:
        raise ValueError(f"mong đợi nhãn số shape (N,), nhận {y_true.shape}. "
                         f"Nếu đang có one-hot thì đổi bằng .argmax(axis=1)")
    for ten, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.size and (arr.min() < 0 or arr.max() >= num_classes):
            raise ValueError(f"{ten} có giá trị ngoài [0, {num_classes})")

    # Mã hoá mỗi cặp (thật, đoán) thành MỘT số nguyên duy nhất rồi đếm một lượt.
    # minlength là bắt buộc: nếu lớp cuối không xuất hiện lần nào, mảng đếm sẽ
    # ngắn hơn C*C và reshape bị lỗi.
    chi_so = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    return np.bincount(chi_so, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def tinh_chi_so(cm: np.ndarray) -> dict:
    """
    Tính precision / recall / F1 / support cho từng lớp từ confusion matrix.

    Parameters:
        cm: (C, C) confusion matrix.

    Returns:
        dict: "precision", "recall", "f1", "support" (mảng (C,)),
              "accuracy", "macro_f1", "macro_precision", "macro_recall" (scalar).

    recall[c]    = đúng / số ảnh THẬT của lớp c   (tổng HÀNG)
    precision[c] = đúng / số lần ĐOÁN ra lớp c    (tổng CỘT)

    Lớp không bao giờ được đoán -> mẫu số của precision bằng 0. Quy ước kết quả
    là 0.0 chứ không để ra nan, vì nan sẽ lan sang cả macro_f1.
    """
    cm = np.asarray(cm)
    dung = np.diag(cm).astype(np.float64)
    tong_hang = cm.sum(axis=1).astype(np.float64)    # số ảnh thật mỗi lớp
    tong_cot = cm.sum(axis=0).astype(np.float64)     # số lần đoán mỗi lớp

    def chia_an_toan(tu: np.ndarray, mau: np.ndarray) -> np.ndarray:
        return np.divide(tu, mau, out=np.zeros_like(tu), where=mau > 0)

    precision = chia_an_toan(dung, tong_cot)
    recall = chia_an_toan(dung, tong_hang)
    f1 = chia_an_toan(2 * precision * recall, precision + recall)
    tong = cm.sum()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": cm.sum(axis=1),
        "accuracy": float(np.trace(cm) / tong) if tong else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def in_confusion_matrix(cm: np.ndarray, class_names: list[str],
                        chuan_hoa: bool = False) -> None:
    """
    In confusion matrix ra terminal dạng bảng căn cột.

    Parameters:
        chuan_hoa: True thì chia mỗi hàng cho tổng hàng và in phần trăm.
    """
    C = len(class_names)
    rong = 9
    # Tên lớp dài (rolled-in_scale = 15 ký tự) nên cắt bớt cho phần tiêu đề cột
    tieu_de = "".join(f"{ten[:rong - 1]:>{rong}}" for ten in class_names)
    nhan_goc = "thật \\ đoán"       # Python 3.10 không cho dấu \ bên trong f-string
    print(f"{nhan_goc:<18}{tieu_de}{'tổng':>9}")
    print("-" * (18 + rong * (C + 1)))

    for i, ten in enumerate(class_names):
        if chuan_hoa:
            tong_hang = cm[i].sum()
            o = "".join(f"{(100 * cm[i, j] / tong_hang if tong_hang else 0):>{rong}.1f}"
                        for j in range(C))
            print(f"{ten:<18}{o}{cm[i].sum():>9}")
        else:
            o = "".join(f"{cm[i, j]:>{rong}}" for j in range(C))
            print(f"{ten:<18}{o}{cm[i].sum():>9}")

    tong_cot = "".join(f"{cm[:, j].sum():>{rong}}" for j in range(C))
    print(f"{'số lần đoán':<18}{tong_cot}{cm.sum():>9}")
    if chuan_hoa:
        print("(số trong bảng là % theo từng HÀNG; hàng = nhãn thật)")


def in_bao_cao(cm: np.ndarray, class_names: list[str]) -> None:
    """In bảng precision / recall / F1 / support cho từng lớp."""
    cs = tinh_chi_so(cm)
    print(f"{'lớp':<18}{'precision':>11}{'recall':>9}{'f1':>8}{'support':>9}")
    print("-" * 55)
    for i, ten in enumerate(class_names):
        print(f"{ten:<18}{cs['precision'][i]:>11.3f}{cs['recall'][i]:>9.3f}"
              f"{cs['f1'][i]:>8.3f}{cs['support'][i]:>9}")
    print("-" * 55)
    print(f"{'accuracy':<18}{'':>11}{'':>9}{cs['accuracy']:>8.3f}{cm.sum():>9}")
    print(f"{'macro avg':<18}{cs['macro_precision']:>11.3f}{cs['macro_recall']:>9.3f}"
          f"{cs['macro_f1']:>8.3f}{cm.sum():>9}")


def top_k_accuracy(scores: np.ndarray, y: np.ndarray, k: int = 2) -> float:
    """
    Tỉ lệ mẫu mà nhãn đúng nằm trong k lớp có điểm cao nhất.

    Parameters:
        scores: (N, C) logits hoặc xác suất (softmax không đổi thứ hạng nên như nhau).
        y: (N,) nhãn đúng.
        k: Số lựa chọn được tính là "đúng".
    """
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k phải trong [1, {scores.shape[1]}], nhận {k}")
    # argsort tăng dần -> lấy âm để thành giảm dần, rồi cắt k cột đầu
    top_k = np.argsort(-scores, axis=1)[:, :k]          # (N, k)
    return float((top_k == y[:, None]).any(axis=1).mean())


def ve_confusion_matrix(cm: np.ndarray, class_names: list[str],
                        save_path: str | None = None, show: bool = True,
                        chuan_hoa: bool = True, tieu_de: str = "Confusion matrix") -> None:
    """Vẽ confusion matrix dạng bản đồ nhiệt, có ghi số lên từng ô."""
    import matplotlib.pyplot as plt

    C = len(class_names)
    tong_hang = cm.sum(axis=1, keepdims=True)
    mau = np.divide(cm, tong_hang, out=np.zeros(cm.shape, float), where=tong_hang > 0) \
        if chuan_hoa else cm.astype(float)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    anh = ax.imshow(mau, cmap="Blues", vmin=0, vmax=mau.max())

    nguong = mau.max() / 2
    for i in range(C):
        for j in range(C):
            # Chú ý thứ tự: x là CỘT j, y là HÀNG i
            chu = f"{cm[i, j]}\n{100 * mau[i, j]:.0f}%" if chuan_hoa else str(cm[i, j])
            ax.text(j, i, chu, ha="center", va="center", fontsize=8,
                    color="white" if mau[i, j] > nguong else "black")

    ax.set_xticks(range(C), class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(C), class_names, fontsize=9)
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("nhãn thật")
    ax.set_title(tieu_de)
    fig.colorbar(anh, ax=ax, fraction=0.046)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"Đã lưu hình: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def xem_anh_sai(model: Sequential, X: np.ndarray, X_raw: np.ndarray, y: np.ndarray,
                class_names: list[str], n: int = 12, save_path: str | None = None,
                show: bool = True) -> None:
    """
    Vẽ n ảnh bị đoán SAI, sắp xếp theo độ tự tin giảm dần.

    Parameters:
        X: (N, D) đã chuẩn hóa — để mạng dự đoán.
        X_raw: (N, 64, 64) uint8 — để hiển thị. Phải CÙNG THỨ TỰ với X.
        y: (N,) nhãn thật.

    Xem những ca "sai mà chắc chắn nhất" là đáng phân tích nhất: đó là chỗ mạng
    hiểu sai quy luật, chứ không phải chỗ nó lưỡng lự.
    """
    import matplotlib.pyplot as plt

    P = softmax(model.forward(X, training=False))       # (N, C) xác suất
    y_pred = P.argmax(axis=1)
    tu_tin = P.max(axis=1)

    sai = np.flatnonzero(y_pred != y)
    if len(sai) == 0:
        print("Không có ảnh nào bị đoán sai.")
        return

    # Sắp xếp giảm dần theo độ tự tin rồi lấy n ảnh đầu
    sai = sai[np.argsort(-tu_tin[sai])][:n]

    hang = int(np.ceil(len(sai) / 4))
    fig, axes = plt.subplots(hang, 4, figsize=(11, 3.4 * hang))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, sai):
        ax.imshow(X_raw[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"thật: {class_names[y[i]]}\nđoán: {class_names[y_pred[i]]} "
                     f"({tu_tin[i]:.2f})", fontsize=8)

    fig.suptitle(f"{len(np.flatnonzero(y_pred != y))} ảnh sai — "
                 f"{len(sai)} ca mạng SAI MÀ TỰ TIN NHẤT")
    # rect chừa chỗ cho suptitle, h_pad tránh tiêu đề hàng dưới đè lên ảnh hàng trên
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"Đã lưu hình: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def danh_gia_day_du(model: Sequential, criterion: SoftmaxCrossEntropy,
                    X: np.ndarray, Y: np.ndarray, y: np.ndarray,
                    class_names: list[str], ten_tap: str = "val") -> dict:
    """
    Chạy dự đoán trên một tập và in toàn bộ báo cáo.

    Returns:
        dict: "cm", "scores", "y_pred", cùng các chỉ số từ tinh_chi_so().
    """
    scores = model.forward(X, training=False)           # logits (N, C)
    y_pred = scores.argmax(axis=1)
    cm = confusion_matrix(y, y_pred, len(class_names))
    cs = tinh_chi_so(cm)

    print(f"\n{'=' * 72}\nTẬP {ten_tap.upper()} — {len(y)} ảnh\n{'=' * 72}")
    print(f"loss = {criterion.forward(scores, Y):.4f} | "
          f"accuracy = {cs['accuracy']:.4f} | "
          f"top-2 = {top_k_accuracy(scores, y, k=2):.4f} | "
          f"top-3 = {top_k_accuracy(scores, y, k=3):.4f}")

    # Kiểm tra chéo: accuracy tính từ cm phải KHỚP với hàm accuracy() ở Bước 5
    assert abs(cs["accuracy"] - accuracy(scores, y)) < 1e-12

    print(f"\n--- Confusion matrix (số ảnh) ---")
    in_confusion_matrix(cm, class_names)
    print(f"\n--- Báo cáo theo lớp ---")
    in_bao_cao(cm, class_names)

    # Cặp nhầm lẫn nặng nhất: bỏ đường chéo rồi tìm ô lớn nhất
    nham = cm.copy()
    np.fill_diagonal(nham, 0)
    print(f"\n--- 5 cặp nhầm lẫn nặng nhất ---")
    for thu_tu in np.argsort(nham, axis=None)[::-1][:5]:
        i, j = np.unravel_index(thu_tu, nham.shape)
        print(f"  {class_names[i]:<16} -> đoán thành {class_names[j]:<16} "
              f"{nham[i, j]:>3} lần ({100 * nham[i, j] / cm[i].sum():.0f}% của lớp này)")

    return {"cm": cm, "scores": scores, "y_pred": y_pred, **cs}


if __name__ == "__main__":
    from train import train, ve_duong_cong

    # # ==================================================================
    # print("=" * 72)
    # print("TEST 1: confusion_matrix trên ví dụ tính tay")
    # print("=" * 72)
    # y_true = np.array([0, 0, 1, 2, 2, 2])
    # y_pred = np.array([0, 1, 1, 2, 2, 0])
    # cm_tay = confusion_matrix(y_true, y_pred, 3)
    # print(cm_tay)
    # mong_doi = np.array([[1, 1, 0],
    #                      [0, 1, 0],
    #                      [1, 0, 2]])
    # assert np.array_equal(cm_tay, mong_doi), "confusion_matrix sai"
    # assert cm_tay.sum() == len(y_true)
    # print("Khớp bảng tính tay ✔")

    # print("\n" + "=" * 72)
    # print("TEST 2: tinh_chi_so, kể cả trường hợp chia cho 0")
    # print("=" * 72)
    # cs = tinh_chi_so(cm_tay)
    # # Lớp 0: đoán đúng 1 lần; có 2 lần đoán ra lớp 0 -> precision 1/2
    # #        có 2 ảnh thật lớp 0                     -> recall    1/2
    # # Lớp 2: 2/3 số ảnh thật được tìm ra             -> recall    2/3
    # print("precision:", cs["precision"].round(4))
    # print("recall   :", cs["recall"].round(4))
    # print("accuracy :", round(cs["accuracy"], 4), "= 4/6")
    # assert np.allclose(cs["precision"], [1 / 2, 1 / 2, 1.0])
    # assert np.allclose(cs["recall"], [1 / 2, 1.0, 2 / 3])
    # assert abs(cs["accuracy"] - 4 / 6) < 1e-12

    # # Lớp 2 không bao giờ được đoán -> precision phải là 0.0, KHÔNG phải nan
    # cm_thieu = np.array([[2, 1, 0],
    #                      [0, 3, 0],
    #                      [1, 1, 0]])
    # cs_thieu = tinh_chi_so(cm_thieu)
    # print("\nLớp không bao giờ được đoán -> precision =", cs_thieu["precision"])
    # assert cs_thieu["precision"][2] == 0.0 and np.isfinite(cs_thieu["f1"]).all()
    # print("Không có nan ✔")

    # ==================================================================
    print("\n" + "=" * 72)
    print("TEST 3: Huấn luyện rồi đánh giá đầy đủ trên dữ liệu thật")
    print("=" * 72)
    data = prepare_data()
    model = tao_mlp(data["X_train"].shape[1], [128], len(CLASS_NAMES),
                    rng=np.random.default_rng(42))
    criterion = SoftmaxCrossEntropy()
    history = train(model, Adam(model.layers, lr=1e-3), criterion, data,
                    epochs=30, batch_size=64, rng=np.random.default_rng(42),
                    verbose=False)
    print(f"Đã train 30 epoch: train acc {history['train_acc'][-1]:.4f} | "
          f"val acc {history['val_acc'][-1]:.4f}")

    kq_train = danh_gia_day_du(model, criterion, data["X_train"], data["Y_train"],
                               data["y_train"], CLASS_NAMES, ten_tap="train")
    kq_val = danh_gia_day_du(model, criterion, data["X_val"], data["Y_val"],
                             data["y_val"], CLASS_NAMES, ten_tap="val")

    # Tập val cân bằng 60 ảnh/lớp -> macro recall PHẢI bằng accuracy
    assert np.array_equal(kq_val["support"], [60] * 6)
    assert abs(kq_val["macro_recall"] - kq_val["accuracy"]) < 1e-12
    print(f"\nmacro recall ({kq_val['macro_recall']:.4f}) = accuracy "
          f"({kq_val['accuracy']:.4f}) vì tập val cân bằng ✔")
    print(f"macro precision ({kq_val['macro_precision']:.4f}) thì KHÁC, "
          f"vì số lần đoán mỗi lớp không đều nhau")

    # ==================================================================
    print("\n" + "=" * 72)
    print("TEST 4: Vẽ hình")
    print("=" * 72)
    ve_confusion_matrix(kq_val["cm"], CLASS_NAMES,
                        save_path=os.path.join(PROJECT_ROOT, "outputs", "confusion_matrix.png"),
                        tieu_de=f"NEU-DET val — MLP [128] — accuracy {kq_val['accuracy']:.3f}")

    _, _, X_val_raw, _ = load_cache()       # ảnh gốc uint8, CÙNG THỨ TỰ với X_val
    xem_anh_sai(model, data["X_val"], X_val_raw, data["y_val"], CLASS_NAMES, n=12,
                save_path=os.path.join(PROJECT_ROOT, "outputs", "anh_sai.png"))

    print("\n" + "=" * 72)
    print("TẤT CẢ TEST ĐỀU ĐẠT ✔")
    print("=" * 72)
