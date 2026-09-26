import numpy as np


# Lớp cơ sở (Base Layer) cho các lớp mạng nơ-ron
class Layer:
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Lan truyền tiến qua lớp.

        Parameters:
            X: Dữ liệu đầu vào, shape (batch, in_features).
            training: True khi huấn luyện, False khi dự đoán.
                Chưa dùng tới, để dành cho Dropout / BatchNorm sau này.

        Returns:
            Đầu ra của lớp.
        """
        raise NotImplementedError("Phương thức forward() chưa được triển khai.")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Lan truyền ngược qua lớp.

        Parameters:
            dout: Gradient của loss theo đầu ra của lớp.

        Returns:
            Gradient của loss theo đầu vào của lớp.
        """
        raise NotImplementedError("Phương thức backward() chưa được triển khai.")

    def params(self) -> dict[str, np.ndarray]:
        """
        Các tham số học được, ví dụ {"W": ..., "b": ...}.
        Lớp không có tham số (như ReLU) trả về {}.
        """
        return {}

    def grads(self) -> dict[str, np.ndarray]:
        """
        Gradient tương ứng với params(), cùng khoá, cùng shape.
        """
        return {}


# Lớp kết nối đầy đủ (DENSE / LINEAR)
class Dense(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = "he",
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ):
        # dtype: mặc định float32 cho nhanh và nhẹ. Khi kiểm tra gradient bằng
        # sai phân số (gradcheck) thì truyền np.float64, vì float32 chỉ có ~7
        # chữ số nên phép trừ hai loss gần bằng nhau sẽ mất hết độ chính xác.
        if rng is None:
            rng = np.random.default_rng()

        # 1. Độ lệch chuẩn theo phương pháp khởi tạo
        if init == "he":
            # Dành cho ReLU: nhân đôi phương sai để bù phần âm bị cắt bỏ
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            # Xavier/Glorot: cân bằng cả chiều vào lẫn chiều ra
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            raise ValueError(f"Phương pháp khởi tạo '{init}' không được hỗ trợ.")

        # 2. Trọng số và bias — CÙNG kiểu dtype
        self.W = (rng.standard_normal((in_features, out_features)) * std).astype(dtype)
        self.b = np.zeros(out_features, dtype=dtype)

        # 3. Gradient — cùng shape, cùng dtype với tham số
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # 4. Cache đầu vào cho backward
        self.X = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"Dense mong đợi shape (batch, {self.W.shape[0]}), nhận {X.shape}"
            )

        self.X = X
        # Z = X @ W + b
        #   (batch, in) @ (in, out) -> (batch, out)
        #   b shape (out,) được broadcast cho mọi hàng
        return X @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Lan truyền ngược qua Z = X @ W + b.

        Parameters:
            dout: dL/dZ, shape (batch, out) — cùng shape với đầu ra của forward.

        Returns:
            dX = dL/dX, shape (batch, in) — cùng shape với đầu vào.

        Đồng thời ghi self.dW (in, out) và self.db (out,).

        Cách nhớ công thức bằng shape:
            dX  = dout @ W.T    (B, out) @ (out, in) -> (B, in)   ✔
            dW  = X.T  @ dout   (in, B)  @ (B, out)  -> (in, out) ✔
            db  = dout.sum(axis=0)                   -> (out,)    ✔

        db là TỔNG chứ không phải trung bình: b được broadcast cộng vào cả B
        hàng, tức tham gia vào loss B lần, nên gradient là tổng đóng góp của
        tất cả các hàng. (Phép chia cho B đã nằm sẵn trong dZ của loss.)
        """
        if self.X is None:
            raise RuntimeError("Phải gọi forward() trước khi gọi backward()")
        if dout.shape != (self.X.shape[0], self.W.shape[1]):
            raise ValueError(
                f"dout mong đợi shape {(self.X.shape[0], self.W.shape[1])}, nhận {dout.shape}"
            )

        # Ghi TẠI CHỖ bằng [...] để optimizer không mất tham chiếu qua grads()
        self.dW[...] = self.X.T @ dout          # (in, out)
        self.db[...] = dout.sum(axis=0)         # (out,) — KHÔNG keepdims

        return dout @ self.W.T                  # (batch, in)

    def params(self) -> dict[str, np.ndarray]:
        # Trả về tham chiếu (reference), không .copy(), để optimizer sửa tại chỗ
        return {"W": self.W, "b": self.b}

    def grads(self) -> dict[str, np.ndarray]:
        return {"W": self.dW, "b": self.db}

    # Method này giúp dễ đọc object
    def __repr__(self) -> str:
        num_params = self.W.size + self.b.size
        return f"Dense({self.W.shape[0]} -> {self.W.shape[1]}, params={num_params:,})"


if __name__ == "__main__":
    print("--- Test 1: Forward so với tính tay ---")
    d = Dense(2, 3, init="he")
    # Ghi TẠI CHỖ bằng [...] thay vì gán lại, giữ nguyên tham chiếu
    d.W[...] = [[1, 0, -1], [2, 1, 0]]
    d.b[...] = [0.5, 0, 0]

    out = d.forward(np.array([[1, 2], [3, 4]], dtype=np.float32))
    print("out =\n", out)
    # Hàng 1: [1*1 + 2*2, 1*0 + 2*1, 1*(-1) + 2*0] + b = [5.5, 2, -1]
    # Hàng 2: [3*1 + 4*2, 3*0 + 4*1, 3*(-1) + 4*0] + b = [11.5, 4, -3]
    expected = np.array([[5.5, 2, -1], [11.5, 4, -3]], dtype=np.float32)
    assert np.allclose(out, expected), "Sai kết quả forward"
    assert out.dtype == np.float32, f"dtype đầu ra phải là float32, nhận {out.dtype}"
    print("OK: đúng giá trị, đúng dtype float32")

    print("\n--- Test 2: Tích hợp với dữ liệu thật ---")
    try:
        from preprocess import prepare_data
    except ImportError:
        print("Bỏ qua: chưa có preprocess.py")
    else:
        data = prepare_data()
        n_features = data["X_train"].shape[1]
        rng = np.random.default_rng(42)
        layer = Dense(n_features, 128, rng=rng)

        Z = layer.forward(data["X_train"][:64])
        print(layer, Z.shape, Z.dtype)
        print("W std = %.4f (lý thuyết %.4f)" % (layer.W.std(), np.sqrt(2 / n_features)))

    print("\n--- Test 3: Khởi tạo W qua 10 lớp (Exploding / Vanishing) ---")
    rng_exp = np.random.default_rng(42)
    X_init = rng_exp.standard_normal((64, 256)).astype(np.float32)

    for name, std_val in [("std=1.0", 1.0), ("std=0.01", 0.01), ("He", None)]:
        X_out = X_init.copy()

        for _ in range(10):
            layer = Dense(256, 256, rng=rng_exp)
            if std_val is not None:
                layer.W[...] = rng_exp.standard_normal((256, 256)) * std_val

            X_out = np.maximum(0, layer.forward(X_out))  # ReLU tạm

        print(f"{name:10s} -> std đầu ra = {X_out.std():.6e}")