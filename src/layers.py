import numpy as np

import numpy as np

# Lớp cơ sở (Base Layer) cho các lớp mạng nơ-ron
class Layer:
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Thực hiện bước lan truyền tiến (forward pass) qua lớp.

        Parameters:
        X (np.ndarray): Dữ liệu đầu vào cho lớp.

        Returns:
        np.ndarray: Kết quả đầu ra sau khi áp dụng lớp.
        """
        raise NotImplementedError("Phương thức forward() chưa được triển khai.")

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Thực hiện bước lan truyền ngược (backward pass) qua lớp.

        Parameters:
        dY (np.ndarray): Gradient của hàm mất mát theo đầu ra của lớp.

        Returns:
        np.ndarray: Gradient của hàm mất mát theo đầu vào của lớp.
        """
        raise NotImplementedError("Phương thức backward() chưa được triển khai.")

    def params(self) -> dict[str, np.ndarray]:
        """
        Trả về các tham số của lớp (nếu có).

        Returns:
        Trả về dict các tham số học được, ví dụ {"W": ..., "b": ...}	return {} (lớp không có tham số, như ReLU)
        dict[str, np.ndarray]: Từ điển chứa các tham số của lớp.
        """
        return {}

    def grads(self) -> dict[str, np.ndarray]:
        """
        Trả về các gradient của các tham số của lớp (nếu có).

        Returns:
        dict[str, np.ndarray]: Từ điển chứa các gradient của các tham số của lớp.
        """
        return {}


# Lớp kết nối đầy đủ (DENSE / LINEAR)
class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, init: str = "he",rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()

        # 1. Tính toán độ lệch chuẩn (std) theo phương pháp khởi tạo
        if init == "he":
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"Phương pháp khởi tạo '{init}' không được hỗ trợ.")

        # 2. Khởi tạo trọng số W với phân phối chuẩn (normal distribution)
        self.W = (rng.standard_normal((in_features, out_features)) * std).astype(np.float32)
        self.b = np.zeros(out_features)

        # 3. Biến lưu trữ gradient
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # 4. Cache lưu đầu vào X để dùng cho backward
        self.X = None


    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        # Kiểm tra shape để bắt lỗi sớm
        assert X.shape[1] == self.W.shape[0],f"Dense mong đợi {self.W.shape[0]} features, nhận {X.shape[1]}"

        self.X = X
        # Z = X @ W + b
        return X @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Bước 6")

    def params(self) -> dict[str, np.ndarray]:
        # Trả về tham chiếu (Reference), không dùng .copy()
        return {"W": self.W, "b": self.b}

    def grads(self) -> dict[str, np.ndarray]:
        return {"W": self.dW, "b": self.db}

    def __repr__(self) -> str:
        num_params = self.W.size + self.b.size
        return f"Dense({self.W.shape[0]} -> {self.W.shape[1]}, params={num_params:,})"


if __name__ == "__main__":
    print("--- Test 1: Tính toán forward ( so sánh tính tay) ---")
    d = Dense(2, 3, init="he")
    d.W = np.array([[1, 0, -1], [2, 1, 0]], dtype=np.float32)
    d.b = np.array([0.5, 0, 0], dtype=np.float32)

    out = d.forward(np.array([[1, 2], [3, 4]], dtype=np.float32))
    print("out =\n", out)
    # assert np.allclose(out, np.array([[4.5, 2, -1], [8.5, 4, -3]], dtype=np.float32)), "Sai kết quả forward"

    print("-------Tích hợp với data loader-------")
    try:
        from preprocess import prepare_data
        data = prepare_data()
        rng = np.random.default_rng(42)
        layer = Dense(4096, 128, rng=rng)
        
        Z = layer.forward(data["X_train"][:64])
        print(layer, Z.shape, Z.dtype)
        print("W std = %.4f (lý thuyết %.4f)" % (layer.W.std(), np.sqrt(2/4096)))
    except ImportError:
        print("⚠️ Cảnh báo: Không tìm thấy thư viện preprocess hoặc có lỗi khi load data.")

    print("\n--- Test 3: Thí nghiệm khởi tạo W qua 10 lớp (Vấn đề Exploding/Vanishing) ---")
    rng_exp = np.random.default_rng(42)
    X_init = rng_exp.standard_normal((64, 256)).astype(np.float32)
    
    for name, std_val in [("std=1.0", 1.0), ("std=0.01", 0.01), ("He", None)]:
        X_out = X_init.copy()
        
        for _ in range(10):
            l = Dense(256, 256, rng=rng_exp)
            if std_val is not None:
                # Gán đè W để giả lập khởi tạo sai
                l.W = (rng_exp.standard_normal((256, 256)) * std_val).astype(np.float32)
                
            Z_out = l.forward(X_out)
            # Dùng tạm ReLU
            X_out = np.maximum(0, Z_out)
            
        print(f"{name:10s} -> std đầu ra = {X_out.std():.6e}")

