import numpy as np
from layers import Layer


# Class 1 - Hàm kích hoạt ReLU: ReLU(x) = max(0, x)
class ReLU(Layer):
    def __init__(self):
        super().__init__()
        # Nơi lưu vị trí nào có X > 0, để dùng cho backward
        self.mask = None

    def forward(self, X:np.ndarray, training: bool = True) -> np.ndarray:
        """
        Lan truyền tiến: thay mọi phần tử âm bằng 0

        Parameters:
            X: Đầu vào, shape bất kỳ, vsi dụ (B, 128).
            trainning: Không dùng tới (ReLU chạy giống nhau ở mọi chế độ)

        Returns:
            Mảng cùng shape và cùng dtype với X
        """

        # Ghi nhớ chỗ nào dương thì: True = neural bật, False = neural tắt
        self.mask = X > 0

        # Tạo mảng mới, không sửa X gốc
        return np.maximum(0, X)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Lan truyền ngược qua ReLU.

        Parameters:
            dout: dL/d(đầu ra), cùng shape với X đã đưa vào forward.

        Returns:
            dX = dL/dX, cùng shape.

        Đạo hàm của max(0, x) là 1 ở chỗ dương, 0 ở chỗ âm. Nói cách khác:
        chỗ nào nơ-ron BẬT thì gradient đi qua nguyên vẹn, chỗ nào TẮT thì
        gradient bị chặn lại hoàn toàn. Đó cũng là gốc rễ của hiện tượng
        "dying ReLU": nơ-ron luôn âm thì gradient luôn 0, không bao giờ hồi phục.
        """
        if self.mask is None:
            raise RuntimeError("Phải gọi forward() trước khi gọi backward()")
        if dout.shape != self.mask.shape:
            raise ValueError(f"dout mong đợi shape {self.mask.shape}, nhận {dout.shape}")

        # mask là bool: True -> 1, False -> 0
        return dout * self.mask

    def __repr__(self) -> str:
        return "ReLU()"

# Hàm 2 - softmax (hàm thường không phải Layer)
def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Biến logits thành xác suất: mỗi hàng không âm và có tổng bằng 1.

    Parameters:
        Z: Logits, shape (B, C). Số thực bất kỳ, có thể âm, có thể rất lớn.

    Returns:
        P: Xác suất, shape (B, C), cùng dtype với Z.
            Mỗi phần tử nằm trong [0, 1], mỗi hàng có tổng  = 1.
    """
    if Z.ndim != 2:
        raise ValueError(f"softmax mong đợi shape (B, C), nhận {Z.shape}")

    # Trừ max của từng hàng: mọi số mũ <= 0, exp() không bao giờ trà
    Z_shift = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z_shift)
    P = E / E.sum(axis=1, keepdims=True)
    return P


# (Tùy chọn) Class 3 - LeakyReLU: X nếu X > 0, ngược lại alpha * X
class LeakyReLU(Layer):
    def __init__(self, alpha: float=0.01):
        super().__init__()
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha phải nằm trong [0, 1], nhận {alpha}")
        # Độ dốc của phần âm: 0.01 nghĩa là số âm bị thu nhỏ 100 lần chứ không bị xóa
        self.alpha = alpha
        self.mask = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.mask = X > 0
        # Chỗ nào mask True lấy X, chỗ nào False lấy alpha * X
        return np.where(self.mask, X, self.alpha * X)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Đạo hàm là 1 ở phần dương, alpha ở phần âm — nên gradient không bao giờ
        tắt hẳn, và nơ-ron "chết" vẫn còn đường hồi phục.
        """
        if self.mask is None:
            raise RuntimeError("Phải gọi forward() trước khi gọi backward()")
        if dout.shape != self.mask.shape:
            raise ValueError(f"dout mong đợi shape {self.mask.shape}, nhận {dout.shape}")

        return np.where(self.mask, dout, self.alpha * dout)

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    #---------------------------------------
    print("--- Test 1: ReLU ---")
    r = ReLU()
    X = np.array([[-2, -0.5, 0, 0.5, 2]], dtype=np.float32)
    print(r.forward(X))             
    print(r.mask)
    assert np.array_equal(r.forward(X), [[0, 0, 0, 0.5, 2]])
    assert np.array_equal(r.mask, [[False, False, False, True, True]])

    print("\n--- Test 1b: LeakyReLU so với ReLU (cùng đầu vào) ---")
    lr = LeakyReLU(alpha=0.01)
    out_leaky = lr.forward(X)
    print("ReLU     :", r.forward(X))
    print("LeakyReLU:", out_leaky, out_leaky.dtype)
    assert np.allclose(out_leaky, [[-0.02, -0.005, 0, 0.5, 2]])
    assert out_leaky.dtype == np.float32
    print(lr)
    #-----------------------------------------
    print("\n--- Test 2: Softmax tính tay")
    P = softmax(np.array([[1, 2, 3]], dtype=np.float32))
    # print(P)                     # [[0.0900 0.2447 0.6652]]
    # e^1 : e^2 : e^3 = 2.718 : 7.389 : 20.09, tổng 30.19
    assert np.allclose(P, [[0.0900, 0.2447, 0.6652]], atol=1e-4)

    # ------------------------------------------------------------------
    print("\n--- Test 3: Thí nghiệm ổn định số học ---")

    def softmax_naive(Z):
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    inputs = [
        [[1, 2, 3]],
        [[1000, 1001, 1002]],
        [[-1000, -1001, -1002]],
    ]
    print(f"{'Đầu vào':<22}{'naive':<24}stable")
    for z in inputs:
        Z = np.array(z, dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            p_naive = softmax_naive(Z)
        p_stable = softmax(Z)
        print(f"{str(z[0]):<22}{str(p_naive[0].round(3)):<24}{p_stable[0].round(3)}")
        assert np.isfinite(p_stable).all()

    # ------------------------------------------------------------------
    print("\n--- Test 4: Tính chất của softmax trên batch ngẫu nhiên ---")
    rng = np.random.default_rng(0)
    Z = (rng.standard_normal((64, 6)) * 10).astype(np.float32)
    # print("Z =", Z.shape)
    P = softmax(Z)    
    # print("P =", P)
    assert P.shape == (64, 6) and P.dtype == np.float32
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)     # mỗi hàng tổng = 1
    assert np.all(P >= 0)                                 # không âm (có thể = 0 do underflow)
    assert np.allclose(softmax(Z + 50), P, atol=1e-6)     # bất biến khi dịch chuyển
    assert np.array_equal(P.argmax(1), Z.argmax(1))       # không đổi thứ hạng
    print("Không có AssertionError. P nhỏ nhất = %.2e" % P.min())

    # ------------------------------------------------------------------
    print("\n--- Test 5: FORWARD PASS ĐẦU TIÊN CỦA CẢ MẠNG ---")
    from layers import Dense

    try:
        from preprocess import prepare_data
    except ImportError:
        # Chưa có preprocess.py: dùng dữ liệu giả CÙNG SHAPE để vẫn kiểm tra
        # được phần ghép mạng. Accuracy trên dữ liệu giả không có ý nghĩa.
        print("(Chưa có preprocess.py -> dùng dữ liệu GIẢ, shape giống thật)")
        rng_data = np.random.default_rng(123)
        X_batch = rng_data.random((64, 4096), dtype=np.float32)   # giống ảnh đã chia 255
        y_batch = rng_data.integers(0, 6, size=64)
    else:
        data = prepare_data()
        X_batch = data["X_train"][:64]
        y_batch = data["y_train"][:64]

    rng_net = np.random.default_rng(42)
    dense1 = Dense(X_batch.shape[1], 128, init="he", rng=rng_net)
    relu = ReLU()
    dense2 = Dense(128, 6, init="xavier", rng=rng_net)
    print("Mạng:", dense1, "->", relu, "->", dense2, "-> softmax")

    # X (64, 4096) -> Dense -> ReLU -> Dense -> softmax -> P (64, 6)
    Z1 = dense1.forward(X_batch)
    relu_out = relu.forward(Z1)
    Z2 = dense2.forward(relu_out)
    P = softmax(Z2)

    print("P.shape =", P.shape, "| dtype =", P.dtype)
    print("Tổng mỗi hàng = 1 ?", np.allclose(P.sum(axis=1), 1.0, atol=1e-6))
    print("P[:3] =\n", P[:3].round(3))

    pred = P.argmax(axis=1)
    accuracy = (pred == y_batch).mean()
    print("pred[:10] =", pred[:10])
    print("Accuracy  = %.3f  (đoán mò 1/6 = %.3f)" % (accuracy, 1 / 6))
    print("Tỉ lệ số 0 sau ReLU = %.3f" % (relu_out == 0).mean())
    print("P lớn nhất cả batch = %.3f" % P.max())

    assert P.shape == (64, 6)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)

