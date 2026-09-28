"""
Sequential: xâu chuỗi các lớp thành một mạng.

Trước đây phải viết tay cả chuỗi gọi, và thứ tự hai dòng phải khớp nhau:
    Z = d2.forward(relu.forward(d1.forward(X)))
    d1.backward(relu.backward(d2.backward(crit.backward())))
Thêm một lớp là phải sửa cả hai chỗ, rất dễ nhầm. Sequential lo việc đó:
forward chạy XUÔI, backward chạy NGƯỢC.

Lưu ý: SoftmaxCrossEntropy KHÔNG nằm trong Sequential, vì nó nhận 2 đầu vào
(Z và Y) và trả về một số. Vòng lặp train nối hai phần lại bằng:
    model.backward(criterion.backward())
"""

import numpy as np

from activations import ReLU
from layers import BatchNorm1d, Dense, Dropout, Layer


class Sequential:
    """Danh sách các lớp chạy nối tiếp nhau."""

    def __init__(self, layers: list[Layer]):
        if not layers:
            raise ValueError("Sequential cần ít nhất một lớp")
        self.layers = list(layers)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Chạy tiến qua toàn bộ các lớp.

        Parameters:
            X: Đầu vào, shape (B, D).
            training: True khi huấn luyện, False khi đánh giá / dự đoán.
                Từ Bước 10 (Dropout, BatchNorm) hai chế độ sẽ khác nhau thật sự.

        Returns:
            LOGITS, shape (B, C) — chưa qua softmax.
        """
        for layer in self.layers:
            X = layer.forward(X, training)
        return X

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Lan truyền ngược qua toàn bộ các lớp, theo thứ tự ĐẢO NGƯỢC.

        Parameters:
            dout: dL/d(logits), shape (B, C) — thường là criterion.backward().

        Returns:
            dL/dX của đầu vào ngoài cùng, shape (B, D).

        reversed() là chỗ BẮT BUỘC phải đúng. Lặp xuôi thì gradient chảy sai
        chiều: mạng vẫn chạy, loss vẫn ra số, nhưng học được rất ít.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn số, shape (B,). Luôn chạy ở chế độ đánh giá."""
        return self.forward(X, training=False).argmax(axis=1)

    def params(self) -> list[np.ndarray]:
        """Danh sách tham chiếu tới mọi tham số học được."""
        return [p for layer in self.layers for p in layer.params().values()]

    def buffers(self) -> list[np.ndarray]:
        """
        Danh sách tham chiếu tới trạng thái không học được (running_mean/var của
        BatchNorm). Phải lưu/khôi phục CÙNG với params, nếu không mô hình sẽ dùng
        trọng số của epoch này với thống kê chuẩn hóa của epoch khác.
        """
        return [b for layer in self.layers for b in layer.buffers().values()]

    def so_tham_so(self) -> int:
        """Tổng số tham số học được của cả mạng."""
        return sum(p.size for p in self.params())

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        chuoi = " -> ".join(repr(layer) for layer in self.layers)
        return f"Sequential({chuoi}, {self.so_tham_so():,} tham số)"


def tao_mlp(in_features: int, hidden: list[int], num_classes: int,
            rng: np.random.Generator | None = None,
            dropout: float = 0.0, batchnorm: bool = False) -> Sequential:
    """
    Dựng một MLP: Dense [-> BatchNorm] -> ReLU [-> Dropout] -> ... -> Dense.

    Parameters:
        in_features: Số chiều đầu vào (4096 với ảnh 64x64).
        hidden: Kích thước các lớp ẩn, ví dụ [128] hoặc [256, 128].
            Để [] thì mạng chỉ còn MỘT Dense duy nhất (hồi quy softmax tuyến tính),
            dùng làm mốc so sánh xem lớp ẩn có đáng giá không.
        num_classes: Số lớp đầu ra (6).
        rng: Bộ sinh số ngẫu nhiên, để dựng lại đúng cùng bộ trọng số.
        dropout: Xác suất tắt nơ-ron sau mỗi ReLU. 0 = không dùng.
        batchnorm: Chèn BatchNorm1d giữa Dense và ReLU.

    Returns:
        Sequential đã sẵn sàng dùng.

    Các lớp ẩn dùng khởi tạo "he" vì phía sau có ReLU cắt mất một nửa tín hiệu.
    Lớp cuối dùng "xavier" vì phía sau nó chỉ còn softmax, không có ReLU.

    Thứ tự Dense -> BatchNorm -> ReLU -> Dropout là quy ước phổ biến nhất:
    chuẩn hóa TRƯỚC khi cắt phần âm, và tắt nơ-ron SAU khi đã kích hoạt.
    KHÔNG đặt BatchNorm hay Dropout sau lớp Dense cuối: đầu ra đó là logits,
    phải đi thẳng vào softmax.
    """
    if rng is None:
        rng = np.random.default_rng()

    layers: list[Layer] = []
    truoc = in_features
    for h in hidden:
        layers.append(Dense(truoc, h, init="he", rng=rng))
        if batchnorm:
            layers.append(BatchNorm1d(h))
        layers.append(ReLU())
        if dropout > 0:
            layers.append(Dropout(dropout, rng=rng))
        truoc = h
    layers.append(Dense(truoc, num_classes, init="xavier", rng=rng))

    return Sequential(layers)


def tao_cnn(in_channels: int = 1, img_size: int = 64, num_classes: int = 6,
            kenh: list[int] = (8, 16), hidden: int = 64,
            rng: np.random.Generator | None = None,
            dropout: float = 0.5, batchnorm: bool = False) -> Sequential:
    """
    Dựng CNN: [Conv -> ReLU -> MaxPool] xN -> Flatten -> Dense -> ReLU -> Dropout -> Dense.

    Parameters:
        in_channels: Số kênh ảnh vào (1 với ảnh xám).
        img_size: Cạnh ảnh vuông (64).
        num_classes: Số lớp đầu ra.
        kenh: Số bộ lọc của từng khối conv, ví dụ (8, 16).
        hidden: Số nơ-ron của lớp Dense ẩn sau Flatten.
        dropout: Xác suất tắt nơ-ron ở lớp Dense ẩn.
        batchnorm: Chèn BatchNorm1d sau lớp Dense ẩn.

    Returns:
        Sequential nhận đầu vào (N, in_channels, img_size, img_size).

    Mỗi MaxPool(2) chia đôi cạnh ảnh, nên sau len(kenh) khối thì cạnh còn
    img_size // 2**len(kenh). padding = 1 với cửa sổ 3x3 giữ nguyên kích thước
    trước khi pool ("same"), nên phép tính kích thước rất dễ theo dõi.
    """
    from conv import Conv2D, Flatten, MaxPool2D

    if rng is None:
        rng = np.random.default_rng()

    layers: list[Layer] = []
    truoc = in_channels
    canh = img_size
    for c in kenh:
        layers.append(Conv2D(truoc, c, 3, stride=1, padding=1, rng=rng))
        layers.append(ReLU())
        layers.append(MaxPool2D(2))
        truoc = c
        canh //= 2

    layers.append(Flatten())
    layers.append(Dense(truoc * canh * canh, hidden, init="he", rng=rng))
    if batchnorm:
        layers.append(BatchNorm1d(hidden))
    layers.append(ReLU())
    if dropout > 0:
        layers.append(Dropout(dropout, rng=rng))
    layers.append(Dense(hidden, num_classes, init="xavier", rng=rng))

    return Sequential(layers)


if __name__ == "__main__":
    from losses import SoftmaxCrossEntropy
    from preprocess import prepare_data

    rng = np.random.default_rng(42)

    print("--- Dựng mạng ---")
    for hidden in ([], [128], [256, 128]):
        print(f"hidden={str(hidden):<12}", tao_mlp(4096, hidden, 6, rng=np.random.default_rng(0)))

    print("\n--- Chạy thử trên dữ liệu thật ---")
    data = prepare_data()
    idx = rng.choice(len(data["X_train"]), 64, replace=False)
    Xb, Yb, yb = data["X_train"][idx], data["Y_train"][idx], data["y_train"][idx]

    model = tao_mlp(4096, [128], 6, rng=np.random.default_rng(42))
    crit = SoftmaxCrossEntropy()

    Z = model.forward(Xb)
    loss = crit.forward(Z, Yb)
    dX = model.backward(crit.backward())
    print(f"logits {Z.shape} | loss {loss:.4f} (ln6 = {np.log(6):.4f}) | dX {dX.shape}")
    print("predict:", model.predict(Xb)[:10], "| nhãn thật:", yb[:10])

    assert Z.shape == (64, 6) and dX.shape == Xb.shape
    assert model.so_tham_so() == 4096 * 128 + 128 + 128 * 6 + 6

    # forward/backward của Sequential phải khớp với chuỗi gọi tay
    d1, relu, d2 = model.layers
    Z_tay = d2.forward(relu.forward(d1.forward(Xb)))
    assert np.array_equal(Z, Z_tay), "Sequential.forward khác chuỗi gọi tay"

    # Gradient phải khác 0 ở MỌI lớp (nếu backward lặp sai chiều sẽ lộ ra ở đây)
    for i, layer in enumerate(model.layers):
        for ten, g in layer.grads().items():
            assert np.abs(g).max() > 0, f"lớp {i} ({layer}) có gradient {ten} toàn 0"
    print("Gradient khác 0 ở mọi lớp ✔ | forward khớp chuỗi gọi tay ✔")
    print("\nTEST ĐẠT ✔")
