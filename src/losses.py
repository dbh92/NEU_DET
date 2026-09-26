import warnings

import numpy as np

from activations import softmax


# Hàm 1 - log_softmax: tính log P thẳng từ logits, không đi qua P
def log_softmax(Z: np.ndarray) -> np.ndarray:
    """
    Tính log(softmax(Z)) theo cách ổn định số học (log-sum-exp).

    Parameters:
        Z: Logits, shape (B, C).
    
    Returns: 
        log P, shape (B, C), cùng dtype với Z. Mọi phần tử <= 0.
        Không bao giờ ra -inf, kể cả khi logits chênh lệch rất lớn.
    """

    if Z.ndim != 2:
        raise ValueError(f"log_softmax mong đợi shape (B, C), nhận {Z.shape}")

    # Giống softmax: trừ max để mọi số mũ <= 0
    Z_shift = Z - Z.max(axis=1, keepdims=True)  # (B, C), mọi phần tử <= 0

    # log P = (z -m) - log(tổng e^(z-m)); tổng >= 1 nên log không bao giờ ra -inf
    return Z_shift - np.log(np.exp(Z_shift).sum(axis=1, keepdims=True))


def cross_entropy(P: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Loss trung bình trên batch, tính từ xác suất đã có sẵn

    Parameters:
        P: Xác suất (đầu ra của softmax), shape (B, C).
        Y: Nhãn đúng dạng one-hot, shape (B, C).
        eps: Chặn dưới khi lấy log, tránh log(0) = -inf.
    
    Returns:
        float: loss trung bình của batch.

    Lưu ý: loss bị chặn trên ở -log(eps) = 27.63. Khi huấn luyện thật,
    dùng SoftmaxCrossEntropy (Cách B) để không bị chặn.
    """ 
    if P.shape != Y.shape:
        raise ValueError(f"P và Y phải cùng shape, nhận {P.shape} và {Y.shape}")

    batch_size = P.shape[0]

    # clip KẸP giá trị vào [eps, 1.0] trước khi lấy log
    # Nếu không kẹp: P = 0 cho ra log(0) = -inf, và ở những cột mà Y = 0
    # thì 0 * (-inf) = nan, làm hỏng loss của cả batch.
    logP = np.log(np.clip(P, eps, 1.0))

    # Y là one-hot nên phép nhân này lọc ra đúng xác suất của lớp đúng;
    # Các cột còn lại nhân 0 nên biến mất.
    # Cộng 0.0 để tránh in ra "-0.0000" khi loss bằng 0.
    return float(-np.sum(Y * logP) / batch_size) + 0.0

    
class SoftmaxCrossEntropy:
    """
    Gộp Softmax và Cross-Entropy thành một khối duy nhất.
    
    Không kế thừa Layer vì nó khác mọi lớp khác ở hai điểm:
    Nhận 2 đầu vào (Z và Y), và trả về 1 số chứ không phải mảng

    Gộp lại có 2 lợi ích:
        1. Ổn định số học: dùng log_softmax nên không bao giờ ra -inf, 
            loss không bị chặn trần như cách clip.
        2. Gradient rút gọn: Ở bước 6 sẽ thấy nó chỉ còn (P - Y)/ B
    """

    def __init__(self):
        # Lưu lại cho backward ở bước 6
        self.P = None   # xác suất, (B, C)
        self.Y = None   # nhãn one-hot, (B, C)

    def forward(self, Z: np.ndarray, Y: np.ndarray) -> float:
        """
        Parameters:
            Z: LOGITS, shape (B, C) - đầu ra thô của Dense cuối cùng, chưa qua softmax
            Y: Nhãn one-hot, shape (B, C)

        Returns: 
            float: loss trung bình của batch.
        """
        if Z.shape != Y.shape:
            raise ValueError(f"Z và Y phải cùng shape, nhận {Z.shape} và {Y.shape}")

        # Cảnh báo (không chặn) nếu Z trông như đã qua softmax
        if Z.size and (Z >= 0).all() and np.allclose(Z.sum(axis=1), 1.0, atol=1e-4):
            warnings.warn(
                "Z trông giống XÁC SUẤT (không âm, mỗi hàng tổng = 1) chứ không phải LOGITS. "
                "Có phải bạn đã lỡ cho qua softmax rồi không? Hãy truyền thẳng đầu ra của Dense cuối.",
                RuntimeWarning, stacklevel=2,
            )
        self.P = softmax(Z)
        self.Y = Y

        batch_size = Z.shape[0]
        # Dùng log_softmax thay cho log(clip(P)): chính xác tuyệt đối, không có trần
        return float(-np.sum(Y * log_softmax(Z)) / batch_size) + 0.0

    def backward(self) -> np.ndarray:
        """
        Gradient của loss theo LOGITS Z.

        Returns:
            dZ, shape (B, C), cùng shape với Z.

        Suy luận: với một mẫu, L = -sum_c Y_c * log(P_c) và P = softmax(z).
        Đạo hàm của riêng softmax là ma trận Jacobian C x C khá rườm rà:
            dP_k/dz_j = P_k * (delta_kj - P_j)
        Nhưng khi ghép với cross-entropy thì mọi thứ triệt tiêu, chỉ còn:
            dL/dz_j = P_j - Y_j        "xác suất đoán trừ đi sự thật"
        Chia thêm cho B vì loss là TRUNG BÌNH của B mẫu.

        Đây chính là lý do gộp Softmax và Cross-Entropy làm một khối:
        không phải dựng ma trận Jacobian C x C cho từng mẫu.
        """
        if self.P is None or self.Y is None:
            raise RuntimeError("Phải gọi forward() trước khi gọi backward()")

        batch_size = self.P.shape[0]
        return (self.P - self.Y) / batch_size

    def __repr__(self) -> str:
        return "SoftmaxCrossEntropy()"


def accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    """
    Tỉ lệ dự đoán đúng, giá trị trong [0, 1].

    Parameters:
        scores: Logits hoặc xác suất, shape (B, C). Dùng cái nào cũng ra kết quả giống nhau, vì softmax không làm đổi thứ hạng.
        y: Nhãn đúng dạng số nguyên , shape (B,) - không phải one-hot

    Returns:
        float trong [0, 1]. Batch rỗng trả về 0.0.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores mong đợi shape (B, C), nhận {scores.shape}")
    if y.ndim != 1:
        raise ValueError(
            f"y phải là nhãn số nguyên shape (B,), nhận {y.shape}. "
            f"Nếu đang có one-hot thì đổi bằng y.argmax(axis=1)."
        )
    if len(scores) != len(y):
        raise ValueError(f"Số mẫu không khớp: {len(scores)} và {len(y)}")

    if len(y) == 0:
        return 0.0      # tránh chia cho 0 -> nan

    # argmax lấy VỊ TRÍ của điểm cao nhất mỗi hàng, chính là lớp được chọn
    predictions = scores.argmax(axis=1) 

    # So sánh ra mảng True/False; True = 1, False = 0 nên mean() là tỉ lệ đúng
    return float((predictions == y).mean())


def test_luong_tinh_loss(batch_size: int = 256, seed: int = 42) -> None:
    """
    Test một luồng duy nhất: cache -> tiền xử lý -> mạng -> loss -> accuracy.

    Đi qua toàn bộ hàm của file này (log_softmax, cross_entropy,
    SoftmaxCrossEntropy, accuracy) trên dữ liệu NEU-DET thật lấy từ
    data_loader.load_cache().

    Parameters:
        batch_size: Số ảnh lấy ngẫu nhiên từ tập train để tính loss.
        seed: Hạt giống ngẫu nhiên, để chạy lại ra đúng cùng kết quả.
    """
    from data_loader import CLASS_NAMES, load_cache
    from preprocess import flatten, fit_standardizer, apply_standardizer, one_hot
    from layers import Dense
    from activations import ReLU

    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(seed)
    C = len(CLASS_NAMES)

    # --- 1. Lấy dữ liệu thô từ cache (tự build cache nếu chưa có) ------
    X_train_raw, y_train, X_val_raw, y_val = load_cache()
    print(f"1. Cache      : X_train {X_train_raw.shape} {X_train_raw.dtype} "
          f"[{X_train_raw.min()}..{X_train_raw.max()}] | {C} lớp")

    # --- 2. Tiền xử lý: duỗi phẳng + chuẩn hóa (mean/std CHỈ lấy từ train)
    X_train = flatten(X_train_raw)
    mean, std = fit_standardizer(X_train)
    X_train = apply_standardizer(X_train, mean, std)
    Y_train = one_hot(y_train, C)
    print(f"2. Tiền xử lý : X {X_train.shape} {X_train.dtype} "
          f"mean={X_train.mean():.4f} std={X_train.std():.4f} | Y {Y_train.shape}")

    # --- 3. Lấy 1 batch NGẪU NHIÊN --------------------------------------
    # Dữ liệu xếp theo thứ tự lớp, nên X_train[:256] sẽ toàn lớp 0 (crazing).
    idx = rng.choice(len(X_train), batch_size, replace=False)
    Xb, Yb, yb = X_train[idx], Y_train[idx], y_train[idx]
    print(f"3. Batch      : {batch_size} ảnh, phân bố lớp {np.bincount(yb, minlength=C)}")

    # --- 4. Mạng 2 lớp CHƯA huấn luyện: 4096 -> 128 -> 6 ----------------
    dense1 = Dense(X_train.shape[1], 128, init="he", rng=rng)
    relu = ReLU()
    dense2 = Dense(128, C, init="xavier", rng=rng)
    H = relu.forward(dense1.forward(Xb))
    Z = dense2.forward(H)                       # LOGITS, chưa qua softmax
    print(f"4. Forward    : {dense1} -> {relu} -> {dense2}")
    print(f"               H {H.shape} (tỉ lệ số 0: {(H == 0).mean():.3f}) "
          f"-> logits {Z.shape} std={Z.std():.2f}")

    # --- 5. softmax và log_softmax phải khớp nhau -----------------------
    P = softmax(Z)                              # xác suất
    logP = log_softmax(Z)                       # log xác suất, tính thẳng từ logits
    assert np.allclose(np.exp(logP), P, atol=1e-6), "exp(log_softmax) phải bằng softmax"
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), "mỗi hàng P phải có tổng = 1"
    assert np.all(logP <= 0), "log của xác suất luôn <= 0"
    print(f"5. Softmax    : P {P.shape} tổng mỗi hàng = 1 ✔ | "
          f"exp(log_softmax) == softmax ✔ | P[0] = {P[0]}")

    # --- 6. Loss: hai cách phải cho cùng kết quả ------------------------
    loss_A = cross_entropy(P, Yb)               # Cách A: từ xác suất, có clip
    criterion = SoftmaxCrossEntropy()
    loss_B = criterion.forward(Z, Yb)           # Cách B: từ logits, log-sum-exp
    assert np.isclose(loss_A, loss_B, atol=1e-5), f"A={loss_A} khác B={loss_B}"
    assert criterion.P.shape == Z.shape and criterion.Y is Yb   # đã lưu cho Bước 6
    print(f"6. Loss       : Cách A (clip) = {loss_A:.4f} | "
          f"Cách B (log-softmax) = {loss_B:.4f} | ln(6) = {np.log(C):.4f}")

    # --- 7. Accuracy: dùng logits hay xác suất đều như nhau -------------
    acc = accuracy(Z, yb)
    assert acc == accuracy(P, yb), "softmax không đổi thứ hạng nên accuracy phải bằng nhau"
    print(f"7. Accuracy   : {acc:.4f} (đoán mò = {1 / C:.4f}) | dự đoán 10 ảnh đầu: {Z.argmax(axis=1)[:10]}")

    # --- 8. Kết luận: mạng chưa học nên loss phải quanh ln(6) -----------
    # Cao hơn ln(6) một chút là ĐÚNG: mạng đoán ngẫu nhiên nhưng lại tự tin
    # (logits std ~1.6). Nhân nhỏ logits lại thì loss tiến đúng về ln(6).
    loss_khiem_ton = criterion.forward((Z * 0.01).astype(np.float32), Yb)
    assert abs(loss_khiem_ton - np.log(C)) < 1e-2
    assert 1.5 < loss_B < 3.0, f"loss ban đầu {loss_B:.2f} bất thường: xem lại chuẩn hóa/khởi tạo"
    assert 0.03 < acc < 0.35, f"accuracy ban đầu {acc:.2f} bất thường"
    print(f"8. Kết luận   : logits thu nhỏ 100 lần -> loss = {loss_khiem_ton:.4f} = ln(6) ✔")
    print("   Mạng chưa huấn luyện: loss quanh ln(6), accuracy quanh 1/6 -> ĐÚNG NHƯ MONG ĐỢI")


if __name__ == "__main__":
    test_luong_tinh_loss()
    print("\nTEST ĐẠT ✔")
