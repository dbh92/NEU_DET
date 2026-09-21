import os
from typing import Iterator

import numpy as np

from data_loader import CLASS_NAMES, PROJECT_ROOT, load_cache


def flatten(X: np.ndarray) -> np.ndarray:
    """
    Flattens a 3D array (N, H, W) into a 2D array (N, H*W).

    Parameters:
    X (np.ndarray): A numpy array of shape (N, H, W).

    Returns:
    np.ndarray: A 2D numpy array of shape (N, H*W) with dtype float32.
    """

    # Lấy số lượng ảnh (N) từ shape của X
    N = X.shape[0]

    # Reshape giữ nguyên chiều N, gộp phần còn lại (-1 sẽ tự tính là H*W)
    # Sau đó ép kiểu về float32
    return X.reshape(N, -1).astype(np.float32)




def fit_standardizer(X: np.ndarray) -> tuple[float, float]:
    """
    Tính toán giá trị trung bình (mean) và độ lệch chuẩn (std) trên TOÀN BỘ mảng tập TRAIN.
    
    Args:
        X (np.ndarray): Mảng dữ liệu đầu vào shape (N, D), dtype float32 (thường là X_train).
        
    Returns:
        tuple[float, float]: Cặp giá trị (mean, std) tính trên toàn bộ các pixel.
    """
    # Tính mean và std trên TOÀN BỘ mảng (không truyền axis) -> 1 cặp số dùng chung cho mọi pixel
    mean_val = float(X.mean())
    std_val = float(X.std())

    return mean_val, std_val




def apply_standardizer(X: np.ndarray, mean: float, std: float, eps: float = 1e-8) -> np.ndarray:
    """
    Chuẩn hóa mảng X dựa trên giá trị mean và std (thường được tính từ tập huấn luyện Train).

    Parameters:
    X (np.ndarray): A 2D numpy array to be standardized.
    mean (float): The mean value for standardization.
    std (float): The standard deviation value for standardization.
    eps (float): Hằng số nhỏ để tránh chia cho 0 nếu std = 0. Default is 1e-8.

    Returns:
    np.ndarray: A standardized 2D numpy array.
    """

    # Với NumPy 2 (NEP 50), float Python là "weak scalar" nên kết quả vẫn là float32.
    # Nhưng nếu mean/std là np.float64 thì kết quả bị đẩy lên float64 -> vẫn ép kiểu cho chắc.
    X_scaled = (X - mean) / (std + eps)

    # Ép kiểu về float32 để tránh lỗi khi huấn luyện mô hình.   
    return X_scaled.astype(np.float32)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Chuyển đổi nhãn y thành dạng one-hot encoding.
    Chuyển đổi mảng nhãn 1D thành mảng nhãn 2D với one-hot encoding.

    Parameters:
    y (np.ndarray): Mảng nhãn 1D (N,) chứa các giá trị nhãn từ 0 đến num_classes-1.
    num_classes (int): Số lượng lớp (classes) trong bài toán phân loại.

    Returns:
    np.ndarray: Ma trận shape (N, num_classes), dtype=np.float32 với one-hot encoding của nhãn y.
    """
    # Tạo mảng one-hot với kích thước (số lượng mẫu, số lượng lớp)
    # Khuyên dùng, khởi tạo ma trận toàn số 0 với shape (N, num_classes)
    N = y.shape[0]
    Y = np.zeros((N, num_classes), dtype=np.float32)
    
    # Gán giá trị 1 vào vị trí tương ứng của nhãn
    # Dùng fancy indexing: Tại hàng thứ i, bật cột thứ y[i] lên 1
    Y[np.arange(N), y] = 1.0

    return Y  


def shuffle_data(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Xáo trộn đồng bộ dữ liệu (X) và nhãn (y) bằng cùng một hoán vị ngẫu nhiên.

    Parameters:
        X (np.ndarray): Mảng dữ liệu đầu vào, shape (N, ...).
        y (np.ndarray): Mảng nhãn tương ứng, shape (N, ...).
        rng (np.random.Generator): Bộ sinh số ngẫu nhiên.

    Returns:
        tuple[np.ndarray, np.ndarray]: Cặp (X, y) đã được xáo trộn đồng bộ.
    """
    # Lấy số lượng mẫu dữ liệu (N) từ chiều đầu tiên của X
    N = X.shape[0]

    idx = rng.permutation(N)  # Tạo một hoán vị ngẫu nhiên của các chỉ số từ 0 đến N-1

    # Dùng fancey indexing để xáo trộn X và y theo cùng một hoán vị idx
    return X[idx], y[idx]

# Mỗi lần duyệt hết generator này = 1 epoch
def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, rng: np.random.Generator | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Tạo các minibatch từ dữ liệu X và nhãn y cho quá trình huấn luyện mô hình.

    Parameters:
        X (np.ndarray): Mảng dữ liệu đầu vào, shape (N, D).
        y (np.ndarray): Mảng nhãn (one-hot hoặc label), shape (N, C) hoặc (N,).
        batch_size (int): Số lượng mẫu trong mỗi batch.
        shuffle (bool): Có xáo trộn dữ liệu ở mỗi epoch hay không.
        rng (np.random.Generator | None): Bộ sinh số ngẫu nhiên.

    Yields:
        tuple[np.ndarray, np.ndarray]: Cặp (X_batch, y_batch) trích xuất từ X và y.
    """
    N = X.shape[0]

    # Tạo mảng chỉ số (index)
    if shuffle:
        # Nếu yêu cầu shuffle mà quên truyền rng, ta tự tạo một bộ mặc định
        if rng is None:
            rng = np.random.default_rng()

        idx = rng.permutation(N)  # Tạo hoán vị ngẫu nhiên
    else:
        idx = np.arange(N)  # Không xáo trộn, giữ nguyên thứ tự

    # Duyệt qua các mẫu với bước nhẩy là batch_size
    for start in range(0, N, batch_size):
        # Start + batch_size có thể vượt quá N, nhưng Numpy sẽ tự động cắt phần dư một cách an toàn
        b = idx[start:start + batch_size]
        yield X[b], y[b]

def prepare_data() -> dict:
    """
    Pipeline tiền xử lý toàn bộ dữ liệu: Load -> Flatten -> Scale -> One-hot.
    Không shuffle ở đây: iterate_minibatches sẽ shuffle lại ở mỗi epoch.

    Returns:
        dict: X_train (N, D) float32, Y_train (N, C) float32, y_train (N,) int64,
              X_val, Y_val, y_val tương tự, và mean/std (float) của tập train.
    """
    # 1. Load data từ cache (load_cache tự build nếu chưa có)
    X_train, y_train, X_val, y_val = load_cache()

    # 2. Flatten cả train và val (từ 3D -> 2D)
    X_train_flat = flatten(X_train)
    X_val_flat = flatten(X_val)

    # 3. Fit standardizer chỉ trên tập train
    mean_val, std_val = fit_standardizer(X_train_flat)

    # 4. Apply standardizer cho cả train và val
    X_train_scaled = apply_standardizer(X_train_flat, mean_val, std_val)
    X_val_scaled = apply_standardizer(X_val_flat, mean_val, std_val)

    # 5. One-hot encoding cho nhãn
    y_train_onehot = one_hot(y_train, num_classes=len(CLASS_NAMES))
    y_val_onehot = one_hot(y_val, num_classes=len(CLASS_NAMES))

    # Trả về dữ liệu đã được tiền xử lý
    return {
        "X_train": X_train_scaled,
        "Y_train": y_train_onehot,
        "y_train": y_train, # Giữ lại để tính accuracy sau này
        "X_val": X_val_scaled,
        "Y_val": y_val_onehot,
        "y_val": y_val,
        "mean": mean_val,
        "std": std_val
    }


def show_samples(X: np.ndarray, y: np.ndarray, n_per_class: int = 5,
                 rng: np.random.Generator | None = None,
                 save_path: str | None = None, show: bool = True) -> None:
    """
    Vẽ lưới ảnh mẫu: mỗi hàng là 1 lớp, mỗi cột là 1 ảnh chọn ngẫu nhiên từ lớp đó.

    Args:
        X (np.ndarray): Ảnh gốc CHƯA flatten, shape (N, H, W), dtype uint8.
        y (np.ndarray): Nhãn số, shape (N,).
        n_per_class (int): Số ảnh hiển thị cho mỗi lớp.
        rng (np.random.Generator | None): Bộ sinh số ngẫu nhiên để chọn ảnh.
        save_path (str | None): Nếu có thì lưu hình ra file .png.
        show (bool): Có mở cửa sổ hiển thị hay không.
    """
    import matplotlib.pyplot as plt  # import ở đây: module khác import preprocess không phải nạp matplotlib

    if rng is None:
        rng = np.random.default_rng()

    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(1.6 * n_per_class + 1.2, 1.6 * n_classes))

    for label, name in enumerate(CLASS_NAMES):
        # Chỉ số của tất cả ảnh thuộc lớp này, rồi chọn ngẫu nhiên n_per_class ảnh (không lặp)
        class_idx = np.flatnonzero(y == label)
        chosen = rng.choice(class_idx, size=n_per_class, replace=False)

        for col, i in enumerate(chosen):
            ax = axes[label, col]
            # vmin/vmax cố định để mọi ảnh cùng thang xám (không tự kéo giãn tương phản)
            ax.imshow(X[i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)

    fig.suptitle("NEU-DET: ảnh mẫu theo lớp")
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"Đã lưu hình: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    d = prepare_data()
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"{k:8s} {str(v.shape):14s} {v.dtype}")
    print("mean/std gốc:", d["mean"], d["std"])
    print("train sau chuẩn hóa: mean=%.4f std=%.4f" % (d["X_train"].mean(), d["X_train"].std()))
    print("val   sau chuẩn hóa: mean=%.4f std=%.4f" % (d["X_val"].mean(), d["X_val"].std()))

    # one-hot đúng?
    assert np.all(d["Y_train"].sum(axis=1) == 1)
    assert np.all(d["Y_train"].argmax(axis=1) == d["y_train"])

    # mini-batch
    rng = np.random.default_rng(0)
    sizes = [len(xb) for xb, yb in iterate_minibatches(d["X_train"], d["Y_train"], 64, rng=rng)]
    print("số batch:", len(sizes), "| batch cuối:", sizes[-1], "| tổng:", sum(sizes))

    # Xem ảnh mẫu (dùng ảnh gốc uint8, chưa flatten)
    X_train_raw, y_train_raw, _, _ = load_cache()
    show_samples(X_train_raw, y_train_raw, n_per_class=len(CLASS_NAMES), rng=rng,
                 save_path=os.path.join(PROJECT_ROOT, "outputs", "samples.png"))
