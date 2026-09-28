"""
Mạng tích chập (CNN) viết bằng NumPy, dùng mẹo im2col.

MLP ở các bước trước gắn trọng số với VỊ TRÍ TUYỆT ĐỐI: pixel (10,10) và (50,50)
là hai tham số hoàn toàn khác nhau. Vì vậy một vết xước dịch sang phải vài pixel
đã thành vector đầu vào khác hẳn. Đó là lý do rolled-in_scale (vảy cán rải rác
khắp bề mặt, không có vị trí cố định) có recall 0.000 suốt Bước 9 và 10.

Conv2D dùng CHUNG một bộ lọc nhỏ cho mọi vị trí: học một lần, nhận ra ở khắp nơi.

Từ file này, ảnh giữ dạng 4 chiều (N, C, H, W). Ảnh xám nên C = 1.
"""

import numpy as np

from layers import Layer


def im2col(X: np.ndarray, KH: int, KW: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Duỗi mọi cửa sổ trượt thành các hàng của một ma trận.

    Parameters:
        X: (N, C, H, W).
        KH, KW: Chiều cao / rộng của cửa sổ.
        stride: Bước trượt.
        pad: Số pixel 0 đệm quanh mỗi cạnh.

    Returns:
        (N*OH*OW, C*KH*KW) — mỗi hàng là một cửa sổ đã duỗi phẳng,
        với OH = (H + 2*pad - KH)//stride + 1, tương tự OW.

    Nhờ nó, tích chập thành MỘT phép nhân ma trận: cols @ W_phẳng.T + b.

    Giá phải trả là bộ nhớ: với cửa sổ 3x3, mỗi pixel nằm trong tối đa 9 cửa sổ
    nên cols lớn gấp ~9 lần ảnh gốc. Đổi RAM lấy tốc độ, một cách có chủ đích.
    """
    N, C, H, W = X.shape
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W + 2 * pad - KW) // stride + 1
    if OH <= 0 or OW <= 0:
        raise ValueError(f"Cửa sổ {KH}x{KW} quá lớn cho ảnh {H}x{W} với pad={pad}")

    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    col = np.zeros((N, C, KH, KW, OH, OW), dtype=X.dtype)

    # Vòng lặp chạy theo KÍCH THƯỚC CỬA SỔ (9 lần với 3x3), không theo số vị trí
    # (4096 lần) hay số ảnh. Mỗi lát cắt lấy một lúc MỌI vị trí có cùng offset.
    for i in range(KH):
        for j in range(KW):
            col[:, :, i, j, :, :] = Xp[:, :, i:i + stride * OH:stride,
                                             j:j + stride * OW:stride]

    # (N, C, KH, KW, OH, OW) -> (N, OH, OW, C, KH, KW): hai chiều đầu là "vị trí nào",
    # ba chiều sau là "nội dung cửa sổ"
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)


def col2im(cols: np.ndarray, X_shape: tuple, KH: int, KW: int,
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Gom các cửa sổ trở lại thành ảnh, CỘNG DỒN ở chỗ chồng lấn.

    Parameters:
        cols: (N*OH*OW, C*KH*KW).
        X_shape: Shape gốc (N, C, H, W).

    Returns:
        (N, C, H, W).

    Đây KHÔNG phải hàm nghịch đảo của im2col: col2im(im2col(X)) cho ra X nhân với
    số lần mỗi pixel bị đếm. Nó là nghịch đảo đúng nghĩa CHO GRADIENT: một pixel
    tham gia vào 9 cửa sổ thì nhận gradient từ cả 9 cửa sổ, và quy tắc chuỗi bảo
    phải CỘNG mọi đóng góp lại — đúng như db = dout.sum(axis=0) ở Bước 6.
    """
    N, C, H, W = X_shape
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W + 2 * pad - KW) // stride + 1

    col = cols.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    # Đệm thêm stride-1 để lát cắt cuối không bị tràn ra ngoài
    Xp = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1),
                  dtype=cols.dtype)

    for i in range(KH):
        for j in range(KW):
            # += chứ KHÔNG phải =: chồng lấn thì gradient phải cộng dồn
            Xp[:, :, i:i + stride * OH:stride,
                     j:j + stride * OW:stride] += col[:, :, i, j, :, :]

    return Xp[:, :, pad:pad + H, pad:pad + W]


class Conv2D(Layer):
    """
    Tích chập 2 chiều.

    Về bản chất đây chính là Dense, chỉ thêm hai lớp bọc im2col / col2im:
        Dense:  out = X    @ W.T       dW = dout.T @ X      dX = dout @ W
        Conv2D: out = cols @ W_phẳng.T  dW = dout.T @ cols   dcols = dout @ W_phẳng
    Nắm được điều này là nắm được toàn bộ CNN.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0,
                 rng: np.random.Generator | None = None, dtype=np.float32):
        """
        Parameters:
            in_channels: Số kênh vào (1 với ảnh xám).
            out_channels: Số bộ lọc, cũng là số kênh ra.
            kernel_size: Cạnh cửa sổ vuông.
            stride, padding: Bước trượt và số pixel 0 đệm quanh ảnh.
                padding = kernel_size // 2 giữ nguyên kích thước ảnh ("same").
        """
        if rng is None:
            rng = np.random.default_rng()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KH = self.KW = kernel_size
        self.stride = stride
        self.padding = padding

        # He init cho conv: fan_in là số phần tử mà MỘT nơ-ron đầu ra nhìn thấy,
        # tức toàn bộ cửa sổ trên mọi kênh vào.
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.W = (rng.standard_normal((out_channels, in_channels,
                                       kernel_size, kernel_size)) * std).astype(dtype)
        self.b = np.zeros(out_channels, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.cols = None
        self.X_shape = None

    def kich_thuoc_ra(self, H: int, W: int) -> tuple[int, int]:
        """Kích thước ảnh sau khi tích chập."""
        return ((H + 2 * self.padding - self.KH) // self.stride + 1,
                (W + 2 * self.padding - self.KW) // self.stride + 1)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if X.ndim != 4:
            raise ValueError(f"Conv2D mong đợi (N, C, H, W), nhận {X.shape}")
        if X.shape[1] != self.in_channels:
            raise ValueError(f"Conv2D mong đợi {self.in_channels} kênh, nhận {X.shape[1]}")

        N, _, H, W = X.shape
        OH, OW = self.kich_thuoc_ra(H, W)

        self.X_shape = X.shape
        self.cols = im2col(X, self.KH, self.KW, self.stride, self.padding)

        W_phang = self.W.reshape(self.out_channels, -1)         # (C_out, C_in*KH*KW)
        out = self.cols @ W_phang.T + self.b                    # (N*OH*OW, C_out)

        # (N*OH*OW, C_out) -> (N, OH, OW, C_out) -> (N, C_out, OH, OW)
        return out.reshape(N, OH, OW, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cols is None:
            raise RuntimeError("Phải gọi forward() trước khi gọi backward()")

        # Đưa về cùng dạng "một hàng một vị trí" như cols
        dout_phang = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.db[...] = dout_phang.sum(axis=0)
        self.dW[...] = (dout_phang.T @ self.cols).reshape(self.W.shape)

        dcols = dout_phang @ self.W.reshape(self.out_channels, -1)
        return col2im(dcols, self.X_shape, self.KH, self.KW, self.stride, self.padding)

    def params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    def grads(self) -> dict[str, np.ndarray]:
        return {"W": self.dW, "b": self.db}

    def cau_hinh(self) -> dict:
        return {"loai": "Conv2D",
                "in_channels": int(self.in_channels),
                "out_channels": int(self.out_channels),
                "kernel_size": int(self.KH),
                "stride": int(self.stride),
                "padding": int(self.padding)}

    def __repr__(self) -> str:
        so = self.W.size + self.b.size
        return (f"Conv2D({self.in_channels} -> {self.out_channels}, "
                f"{self.KH}x{self.KW}, stride={self.stride}, pad={self.padding}, "
                f"params={so:,})")


class MaxPool2D(Layer):
    """
    Lấy giá trị lớn nhất trong mỗi ô, thu nhỏ ảnh.

    Tác dụng: giảm kích thước (nhanh hơn, ít tham số hơn ở lớp Dense sau) và tạo
    bất biến nhỏ với dịch chuyển — vết xước lệch 1 pixel vẫn cho cùng kết quả.

    Không có tham số học được: params() và grads() đều trả {} (kế thừa từ Layer).
    """

    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        self.KH = self.KW = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.arg = None
        self.X_shape = None
        self.out_shape = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if X.ndim != 4:
            raise ValueError(f"MaxPool2D mong đợi (N, C, H, W), nhận {X.shape}")

        N, C, H, W = X.shape
        OH = (H - self.KH) // self.stride + 1
        OW = (W - self.KW) // self.stride + 1

        # Gộp C vào N: mỗi kênh được pool độc lập, nên coi như N*C ảnh 1 kênh
        cols = im2col(X.reshape(N * C, 1, H, W), self.KH, self.KW, self.stride)

        self.arg = cols.argmax(axis=1)      # nhớ ô nào thắng, cho backward
        self.X_shape = X.shape
        self.out_shape = (N, C, OH, OW)

        return cols.max(axis=1).reshape(N, C, OH, OW)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.arg is None:
            raise RuntimeError("Phải gọi forward() trước khi gọi backward()")

        N, C, H, W = self.X_shape
        # Chỉ ô THẮNG nhận gradient, các ô khác nhận 0
        dcols = np.zeros((self.arg.size, self.KH * self.KW), dtype=dout.dtype)
        dcols[np.arange(self.arg.size), self.arg] = dout.reshape(-1)

        dX = col2im(dcols, (N * C, 1, H, W), self.KH, self.KW, self.stride)
        return dX.reshape(N, C, H, W)

    def cau_hinh(self) -> dict:
        return {"loai": "MaxPool2D",
                "kernel_size": int(self.KH),
                "stride": int(self.stride)}

    def __repr__(self) -> str:
        return f"MaxPool2D({self.KH}x{self.KW}, stride={self.stride})"


class Flatten(Layer):
    """
    Duỗi (N, C, H, W) thành (N, C*H*W) để nối vào Dense.
    Cầu nối giữa phần tích chập và phần fully-connected.
    """

    def __init__(self):
        self.X_shape = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.X_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.X_shape is None:
            raise RuntimeError("Phải gọi forward() trước khi gọi backward()")
        return dout.reshape(self.X_shape)

    def __repr__(self) -> str:
        return "Flatten()"


# ======================================================================
# KIỂM THỬ
# ======================================================================

def conv_ngay_tho(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                  stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Tích chập viết bằng 4 vòng for lồng nhau — chậm nhưng không thể sai.
    Dùng làm mốc đối chiếu cho bản im2col.
    """
    N, C, H, Wd = X.shape
    F, _, KH, KW = W.shape
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (Wd + 2 * pad - KW) // stride + 1

    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((N, F, OH, OW), dtype=X.dtype)

    for n in range(N):
        for f in range(F):
            for i in range(OH):
                for j in range(OW):
                    cua_so = Xp[n, :, i * stride:i * stride + KH,
                                      j * stride:j * stride + KW]
                    out[n, f, i, j] = np.sum(cua_so * W[f]) + b[f]
    return out


if __name__ == "__main__":
    import time

    np.set_printoptions(precision=3, suppress=True)

    # ------------------------------------------------------------------
    print("=" * 72)
    print("TEST 1: im2col trên ví dụ tính tay")
    print("=" * 72)
    X = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    print("ảnh 3x3:\n", X[0, 0])
    cols = im2col(X, 2, 2)
    print("im2col cửa sổ 2x2, stride 1:\n", cols)
    mong_doi = np.array([[0, 1, 3, 4],
                         [1, 2, 4, 5],
                         [3, 4, 6, 7],
                         [4, 5, 7, 8]], dtype=np.float32)
    assert np.array_equal(cols, mong_doi), "im2col sai thứ tự"
    print("Khớp bảng tính tay ✔")

    # Shape với pad, stride, nhiều kênh
    X3 = np.zeros((2, 3, 8, 8), np.float32)
    for pad, stride, mong in [(1, 1, (2 * 8 * 8, 3 * 9)), (0, 2, (2 * 3 * 3, 3 * 9))]:
        c = im2col(X3, 3, 3, stride, pad)
        print(f"  (2,3,8,8) pad={pad} stride={stride} -> {c.shape} (mong đợi {mong})")
        assert c.shape == mong

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST 2: col2im cộng dồn, KHÔNG phải nghịch đảo của im2col")
    print("=" * 72)
    mot = np.ones((1, 1, 3, 3), np.float32)
    dem = col2im(im2col(mot, 2, 2), mot.shape, 2, 2)
    print("bản đồ số lần mỗi pixel bị đếm:\n", dem[0, 0])
    assert np.array_equal(dem[0, 0], [[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    print("Pixel giữa nằm trong cả 4 cửa sổ -> nhận gradient 4 lần ✔")

    # Cửa sổ không chồng lấn (stride = kernel) thì col2im ĐÚNG là nghịch đảo
    X4 = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    quay_lai = col2im(im2col(X4, 2, 2, stride=2), X4.shape, 2, 2, stride=2)
    assert np.array_equal(quay_lai, X4)
    print("Với stride = kernel (không chồng lấn) thì col2im khôi phục đúng ảnh gốc ✔")

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST 3: Conv2D so với cài đặt ngây thơ 4 vòng for")
    print("=" * 72)
    rng = np.random.default_rng(0)
    for pad, stride in [(0, 1), (1, 1), (1, 2)]:
        Xt = rng.standard_normal((4, 3, 9, 9)).astype(np.float32)
        conv = Conv2D(3, 5, 3, stride=stride, padding=pad, rng=np.random.default_rng(1))
        a = conv.forward(Xt)
        b = conv_ngay_tho(Xt, conv.W, conv.b, stride, pad)
        print(f"  pad={pad} stride={stride}: shape {a.shape} | lệch tối đa {np.abs(a - b).max():.2e}")
        assert np.allclose(a, b, atol=1e-5), "im2col cho kết quả khác cài đặt ngây thơ"
    print("Khớp hoàn toàn ✔")

    Xt = rng.standard_normal((16, 3, 32, 32)).astype(np.float32)
    conv = Conv2D(3, 8, 3, padding=1, rng=rng)
    t0 = time.perf_counter(); conv.forward(Xt); t_im2col = time.perf_counter() - t0
    t0 = time.perf_counter(); conv_ngay_tho(Xt, conv.W, conv.b, 1, 1); t_ngay_tho = time.perf_counter() - t0
    print(f"Tốc độ trên (16,3,32,32): im2col {t_im2col*1000:.1f}ms | "
          f"ngây thơ {t_ngay_tho*1000:.1f}ms | nhanh hơn {t_ngay_tho/t_im2col:.0f} lần")

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST 4: Bộ lọc đơn vị phải trả về đúng ảnh gốc")
    print("=" * 72)
    conv = Conv2D(1, 1, 3, padding=1, rng=rng)
    conv.W[...] = 0
    conv.W[0, 0, 1, 1] = 1.0            # chỉ tâm bằng 1
    conv.b[...] = 0
    anh = rng.standard_normal((2, 1, 5, 5)).astype(np.float32)
    assert np.allclose(conv.forward(anh), anh)
    print("Conv với bộ lọc đơn vị = ánh xạ đồng nhất ✔")

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST 5: MaxPool2D tính tay")
    print("=" * 72)
    X = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    print("ảnh 4x4:\n", X[0, 0])
    pool = MaxPool2D(2)
    out = pool.forward(X)
    print("sau MaxPool 2x2:\n", out[0, 0])
    assert np.array_equal(out[0, 0], [[5, 7], [13, 15]])

    # backward: chỉ ô thắng nhận gradient
    dX = pool.backward(np.ones_like(out))
    print("gradient dội ngược (1 ở ô thắng, 0 ở ô thua):\n", dX[0, 0])
    assert np.array_equal(dX[0, 0], [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]])

    # Nhiều ảnh, nhiều kênh: kết quả phải độc lập từng kênh
    Xm = rng.standard_normal((3, 4, 8, 8)).astype(np.float32)
    om = MaxPool2D(2).forward(Xm)
    assert om.shape == (3, 4, 4, 4)
    for n in range(3):
        for c in range(4):
            assert np.allclose(om[n, c], Xm[n, c].reshape(4, 2, 4, 2).max(axis=(1, 3)))
    print("Pool độc lập từng kênh, đúng trên (3,4,8,8) ✔")

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST 6: Flatten")
    print("=" * 72)
    fl = Flatten()
    Xf = rng.standard_normal((5, 3, 4, 4)).astype(np.float32)
    of = fl.forward(Xf)
    print(f"{Xf.shape} -> {of.shape} -> backward {fl.backward(of).shape}")
    assert of.shape == (5, 48) and fl.backward(of).shape == Xf.shape

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST 7: GRADCHECK cho Conv2D + MaxPool2D + Flatten")
    print("=" * 72)
    from gradcheck import check_gradients_mang
    from activations import ReLU
    from layers import Dense

    rng = np.random.default_rng(0)
    Xg = rng.standard_normal((2, 1, 6, 6))          # float64
    Yg = np.eye(3)[rng.integers(0, 3, 2)]

    layers = [
        Conv2D(1, 2, 3, padding=1, rng=rng, dtype=np.float64),
        ReLU(),
        MaxPool2D(2),
        Flatten(),
        Dense(2 * 3 * 3, 3, init="xavier", rng=rng, dtype=np.float64),
    ]
    check_gradients_mang(layers, Xg, Yg)

    print("\n" + "=" * 72)
    print("TẤT CẢ TEST ĐỀU ĐẠT ✔")
    print("=" * 72)
