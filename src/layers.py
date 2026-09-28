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

    def cau_hinh(self) -> dict:
        """
        Mô tả đủ để DỰNG LẠI lớp này: {"loai": ..., và các tham số của __init__}.

        Dùng khi lưu/nạp mô hình. Giá trị phải lấy từ chính thuộc tính của object
        (ví dụ self.W.shape[0]) chứ đừng lưu thêm biến riêng lúc __init__ — hai
        nguồn sự thật sẽ lệch nhau ngay khi bạn sửa code.

        Lớp không có tham số cấu trúc (ReLU, Flatten) chỉ cần mặc định này.
        """
        return {"loai": type(self).__name__}

    def buffers(self) -> dict[str, np.ndarray]:
        """
        Trạng thái KHÔNG học được bằng gradient nhưng VẪN là một phần của mô hình,
        ví dụ running_mean / running_var của BatchNorm.

        Optimizer không đụng tới chúng, nhưng khi lưu/nạp mô hình hoặc khi khôi
        phục bộ trọng số tốt nhất (early stopping) thì PHẢI mang theo. Quên là
        mô hình khôi phục sẽ dùng trọng số của epoch này với thống kê của epoch khác.
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

    def cau_hinh(self) -> dict:
        return {"loai": "Dense",
                "in_features": int(self.W.shape[0]),
                "out_features": int(self.W.shape[1])}

    # Method này giúp dễ đọc object
    def __repr__(self) -> str:
        num_params = self.W.size + self.b.size
        return f"Dense({self.W.shape[0]} -> {self.W.shape[1]}, params={num_params:,})"


class Dropout(Layer):
    """
    Mỗi bước train tắt ngẫu nhiên một phần nơ-ron, buộc mạng không được phụ
    thuộc vào bất kỳ nơ-ron riêng lẻ nào. Khi ĐÁNH GIÁ thì không tắt ai cả.

    Dùng "inverted dropout": CHIA cho (1 - p) ngay lúc train, nên kỳ vọng đầu ra
    giữ nguyên và lúc đánh giá không phải làm gì.

    Cách cũ (nhân (1-p) lúc đánh giá) bắt mọi nơi dùng mô hình phải nhớ con số p
    và nhớ nhân lại — chỉ cần một chỗ quên là kết quả sai mà không báo lỗi.
    """

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None,
                 seed: int | None = None):
        """
        Parameters:
            p: Xác suất TẮT mỗi nơ-ron. p = 0 nghĩa là không làm gì.
            rng: Bộ sinh ngẫu nhiên dùng để tạo mask.
            seed: Nếu khác None, mỗi lần forward sẽ tạo lại rng từ seed này, nên
                mask GIỐNG HỆT nhau giữa các lần gọi. Chỉ dùng cho gradcheck:
                sai phân số cần f(x+h) và f(x-h) chạy trên cùng một mask.
        """
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p phải nằm trong [0, 1), nhận {p}")
        self.p = p
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mask = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 0.0:
            self.mask = None
            return X                        # đánh giá: KHÔNG làm gì cả

        rng = np.random.default_rng(self.seed) if self.seed is not None else self.rng
        # mask là float (0 hoặc 1/(1-p)), không phải bool: đã gộp luôn phép chia
        self.mask = (rng.random(X.shape) >= self.p).astype(X.dtype) / (1.0 - self.p)
        return X * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return dout                     # forward chạy ở chế độ đánh giá
        return dout * self.mask

    def cau_hinh(self) -> dict:
        # seed và rng KHÔNG lưu: chúng chỉ ảnh hưởng lúc train, còn lúc dự đoán
        # Dropout là ánh xạ đồng nhất.
        return {"loai": "Dropout", "p": float(self.p)}

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class BatchNorm1d(Layer):
    """
    Chuẩn hóa đầu ra của lớp trước về mean 0, std 1 theo TỪNG FEATURE (axis=0),
    rồi cho mạng tự học lại thang đo qua gamma và beta.

        xhat = (x - mu) / sqrt(var + eps)
        out  = gamma * xhat + beta

    Lợi ích: train ổn định hơn, cho phép learning rate lớn hơn, và có hiệu ứng
    chống overfit nhẹ (mỗi mẫu được chuẩn hóa bằng thống kê của cả batch, nên
    đầu ra của nó phụ thuộc vào các mẫu đi cùng -> một dạng nhiễu có ích).

    Khi ĐÁNH GIÁ thì dùng running_mean/running_var tích luỹ được lúc train, chứ
    không dùng thống kê của batch hiện tại.
    """

    def __init__(self, num_features: int, momentum: float = 0.9,
                 eps: float = 1e-5, dtype=np.float32):
        """
        Parameters:
            num_features: Số chiều D của đầu vào (B, D).
            momentum: Hệ số giữ lại của thống kê chạy (0.9 = giữ 90% giá trị cũ).
            eps: Chống chia cho 0 khi một feature có phương sai bằng 0.
        """
        # Tham số học được. gamma = 1 và beta = 0 -> lúc đầu BatchNorm chỉ thuần
        # chuẩn hóa, chưa bóp méo gì; mạng tự học lại thang đo nếu thấy cần.
        self.gamma = np.ones(num_features, dtype=dtype)
        self.beta = np.zeros(num_features, dtype=dtype)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        # Trạng thái KHÔNG học được, chỉ tích luỹ bằng trung bình trượt
        self.running_mean = np.zeros(num_features, dtype=dtype)
        self.running_var = np.ones(num_features, dtype=dtype)

        self.momentum = momentum
        self.eps = eps
        self.xhat = None
        self.istd = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.gamma.shape[0]:
            raise ValueError(f"BatchNorm1d mong đợi (batch, {self.gamma.shape[0]}), nhận {X.shape}")

        if training:
            mu = X.mean(axis=0)             # (D,) trung bình theo TỪNG FEATURE
            var = X.var(axis=0)             # (D,) không phải theo từng mẫu

            self.istd = 1.0 / np.sqrt(var + self.eps)
            self.xhat = (X - mu) * self.istd

            # Tích luỹ thống kê để dùng lúc đánh giá (ghi tại chỗ)
            self.running_mean[...] = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var[...] = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            self.xhat = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.istd = None

        return self.gamma * self.xhat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Công thức đã rút gọn. Đừng tin nó — để gradcheck xác nhận.

        Điểm tinh tế: mu và var đều phụ thuộc vào MỌI mẫu trong batch, nên đổi
        một mẫu sẽ kéo theo đầu ra của tất cả các mẫu khác. Hai số hạng trừ đi
        trong công thức chính là phần đóng góp gián tiếp qua mu và qua var.
        """
        if self.xhat is None:
            raise RuntimeError("Phải gọi forward() trước khi gọi backward()")
        if self.istd is None:
            raise RuntimeError("forward() đã chạy ở chế độ đánh giá, không backward được")

        B = dout.shape[0]
        self.dgamma[...] = (dout * self.xhat).sum(axis=0)
        self.dbeta[...] = dout.sum(axis=0)

        dxhat = dout * self.gamma
        return (self.istd / B) * (B * dxhat
                                  - dxhat.sum(axis=0)
                                  - self.xhat * (dxhat * self.xhat).sum(axis=0))

    def params(self) -> dict[str, np.ndarray]:
        # Khoá KHÔNG phải "W" nên optimizer sẽ không áp weight decay lên chúng
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self) -> dict[str, np.ndarray]:
        return {"gamma": self.dgamma, "beta": self.dbeta}

    def buffers(self) -> dict[str, np.ndarray]:
        # Không học bằng gradient, nhưng quyết định kết quả lúc đánh giá
        return {"running_mean": self.running_mean, "running_var": self.running_var}

    def cau_hinh(self) -> dict:
        return {"loai": "BatchNorm1d",
                "num_features": int(self.gamma.shape[0]),
                "momentum": float(self.momentum),
                "eps": float(self.eps)}

    def __repr__(self) -> str:
        return f"BatchNorm1d({self.gamma.shape[0]})"


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