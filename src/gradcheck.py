"""
Kiểm tra gradient bằng sai phân số (numerical gradient checking).

Backprop viết sai KHÔNG làm chương trình báo lỗi: mạng vẫn chạy, loss vẫn
giảm chậm chạp, và ta sẽ mất nhiều ngày tưởng rằng mình chọn sai learning rate.
File này so gradient giải tích (công thức tự viết) với gradient đo bằng số
trên một mạng tí hon, để khẳng định công thức đúng trước khi train thật.
"""

import numpy as np

from activations import ReLU
from layers import Dense
from losses import SoftmaxCrossEntropy


def numerical_gradient(f, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """
    Ước lượng dL/dx bằng sai phân trung tâm: (f(x+h) - f(x-h)) / (2h).

    Parameters:
        f: Hàm không tham số, trả về loss (float). Nó phải đọc thẳng mảng x
           hiện tại, vì hàm này sửa x tại chỗ rồi gọi f() lại.
        x: Mảng tham số cần tính gradient. Bị sửa tạm rồi khôi phục nguyên trạng.
        h: Bước nhích. Có HAI nguồn sai số ngược chiều nhau:
           - sai số cắt cụt (do công thức xấp xỉ): giảm khi h nhỏ, cỡ O(h^2)
           - sai số làm tròn (do máy): TĂNG khi h nhỏ, vì f(x+h) và f(x-h) gần
             bằng nhau nên phép trừ nuốt mất các chữ số có nghĩa
           Đo thực tế trên mạng tí hon float64 của file này:
             h = 1e-4 -> sai số ~1e-8   (tốt nhất)
             h = 1e-5 -> sai số ~1e-7
             h = 1e-6 -> sai số ~3e-6   (tệ hơn dù h nhỏ hơn!)
           Nên chọn 1e-4, không phải "càng nhỏ càng tốt".

    Returns:
        Mảng cùng shape với x.

    Dùng sai phân TRUNG TÂM (hai phía) vì sai số chỉ O(h^2), trong khi sai phân
    một phía (f(x+h) - f(x)) / h có sai số O(h) — kém hơn hẳn.

    CỰC CHẬM: mỗi phần tử tốn 2 lần forward. Với W (4096, 128) thì cần hơn
    1 triệu lần forward. Chỉ dùng cho mạng tí hon khi debug, không dùng khi train.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])

    while not it.finished:
        i = it.multi_index
        gia_tri_cu = x[i]

        x[i] = gia_tri_cu + h
        f_cong = f()

        x[i] = gia_tri_cu - h
        f_tru = f()

        # PHẢI khôi phục, nếu không tham số bị lệch dần và mọi kết quả sau đều sai
        x[i] = gia_tri_cu

        grad[i] = (f_cong - f_tru) / (2 * h)
        it.iternext()

    return grad


def rel_error(a: np.ndarray, b: np.ndarray, san: float = 1e-8) -> float:
    """
    Sai số tương đối lớn nhất giữa hai mảng: max |a-b| / max(|a| + |b|, san).

    Dùng sai số TƯƠNG ĐỐI chứ không phải tuyệt đối, vì gradient có thể rất nhỏ
    (1e-8) hoặc rất lớn (1e3) — sai lệch 1e-6 là thảm họa với cái đầu nhưng
    không đáng kể với cái sau.

    Tham số `san` là sàn của mẫu số. Cần nó vì có những gradient ĐÚNG BẰNG 0 về
    mặt toán học, và giá trị tính ra chỉ là nhiễu làm tròn cỡ 1e-17. Chia hai số
    rác cho nhau sẽ ra một tỉ lệ vô nghĩa (thậm chí bằng 1.0) và báo SAI oan.
    Ví dụ thật: bias của lớp Dense đứng NGAY TRƯỚC BatchNorm có gradient bằng 0,
    vì BatchNorm trừ đi trung bình nên cộng thêm hằng số b vào mọi hàng không
    làm đầu ra đổi chút nào. (Đó cũng là lý do các thư viện đặt bias=False cho
    lớp đứng trước BatchNorm — b đó hoàn toàn vô dụng.)
    """
    return float(np.max(np.abs(a - b) / np.maximum(san, np.abs(a) + np.abs(b))))


def check_gradients(seed: int = 0, nguong: float = 1e-6) -> None:
    """
    So gradient giải tích với gradient số trên mạng tí hon 5 -> 4 -> 3.

    Parameters:
        seed: Hạt giống ngẫu nhiên.
        nguong: Ngưỡng sai số tương đối tối đa được chấp nhận.

    Dùng float64 (không phải float32) vì sai phân số lấy hiệu của hai loss gần
    bằng nhau rồi chia cho 2h = 2e-5, làm khuếch đại sai số làm tròn. float32
    chỉ có ~7 chữ số nên sẽ "trượt" ngay cả khi công thức hoàn toàn đúng.
    """
    rng = np.random.default_rng(seed)
    B, D, H, C = 7, 5, 4, 3     # batch 7, vào 5, ẩn 4, ra 3

    X = rng.standard_normal((B, D))                     # float64
    y = rng.integers(0, C, B)
    Y = np.eye(C)[y]                                    # one-hot float64

    dense1 = Dense(D, H, init="he", rng=rng, dtype=np.float64)
    relu = ReLU()
    dense2 = Dense(H, C, init="xavier", rng=rng, dtype=np.float64)
    criterion = SoftmaxCrossEntropy()

    def f() -> float:
        """Chạy tiến cả mạng và trả về loss. Đọc thẳng X, W, b hiện tại."""
        Z1 = dense1.forward(X)
        A1 = relu.forward(Z1)
        Z2 = dense2.forward(A1)
        return criterion.forward(Z2, Y)

    # --- Gradient giải tích: chạy tiến rồi lan ngược -----------------------
    loss = f()
    dZ2 = criterion.backward()          # (B, C)
    dH = dense2.backward(dZ2)           # (B, H)
    dZ1 = relu.backward(dH)             # (B, H)
    dX = dense1.backward(dZ1)           # (B, D)

    # PHẢI copy: numerical_gradient sẽ gọi f() lại nhiều lần, và mỗi lần forward
    # sẽ ghi đè self.dW / self.db / self.X của các lớp.
    giai_tich = {
        "dW1": dense1.dW.copy(), "db1": dense1.db.copy(),
        "dW2": dense2.dW.copy(), "db2": dense2.db.copy(),
        "dX": dX.copy(),
    }
    tham_so = {
        "dW1": dense1.W, "db1": dense1.b,
        "dW2": dense2.W, "db2": dense2.b,
        "dX": X,
    }

    print(f"Mạng tí hon: {D} -> {H} -> {C}, batch = {B}, dtype = float64")
    print(f"loss = {loss:.6f}  (ln {C} = {np.log(C):.6f})\n")
    print(f"{'tham số':<8}{'shape':<12}{'sai số tương đối':<20}kết quả")
    print("-" * 56)

    for ten, g_analytic in giai_tich.items():
        g_numeric = numerical_gradient(f, tham_so[ten])
        sai_so = rel_error(g_analytic, g_numeric)
        dat = sai_so < nguong
        print(f"{ten:<8}{str(g_analytic.shape):<12}{sai_so:<20.3e}{'ĐẠT' if dat else 'SAI'}")
        assert dat, (
            f"{ten}: sai số {sai_so:.3e} >= {nguong:.0e}. Xem lại công thức backward.\n"
            f"giải tích:\n{g_analytic}\nsố:\n{g_numeric}"
        )

    print("\nTẤT CẢ GRADIENT ĐỀU ĐÚNG ✔")


def check_gradients_mang(layers: list, X: np.ndarray, Y: np.ndarray,
                         nguong: float = 1e-6, verbose: bool = True) -> None:
    """
    Kiểm tra gradient cho MỘT MẠNG BẤT KỲ (danh sách lớp), kể cả BatchNorm, Dropout.

    Parameters:
        layers: Danh sách lớp, chạy nối tiếp. Nên dùng dtype float64.
        X: (B, D) đầu vào, float64.
        Y: (B, C) nhãn one-hot, float64.
        nguong: Ngưỡng sai số tương đối tối đa.

    Dropout phải được tạo với seed cố định, nếu không mỗi lần f() chạy lại sẽ
    sinh mask khác nhau và sai phân số trở nên vô nghĩa.

    BatchNorm cũng có điểm tinh tế: mỗi lần forward(training=True) lại cập nhật
    running_mean/running_var. Điều đó KHÔNG ảnh hưởng tới gradient (loss lúc train
    chỉ dùng thống kê của batch hiện tại), nên vẫn kiểm tra được bình thường.
    """
    criterion = SoftmaxCrossEntropy()

    def f() -> float:
        A = X
        for layer in layers:
            A = layer.forward(A, training=True)
        return criterion.forward(A, Y)

    # --- Gradient giải tích ------------------------------------------------
    loss = f()
    dout = criterion.backward()
    for layer in reversed(layers):
        dout = layer.backward(dout)
    dX = dout

    # PHẢI copy trước khi gọi numerical_gradient: nó chạy lại f() nhiều lần và
    # mỗi lần forward sẽ ghi đè cache lẫn gradient của các lớp.
    giai_tich, tham_so = {}, {}
    for i, layer in enumerate(layers):
        p, g = layer.params(), layer.grads()
        for ten in p:
            khoa = f"d{ten}{i}"                 # ví dụ dW0, db0, dgamma1, dbeta1
            giai_tich[khoa] = g[ten].copy()
            tham_so[khoa] = p[ten]
    giai_tich["dX"] = dX.copy()
    tham_so["dX"] = X

    if verbose:
        chuoi = " -> ".join(repr(l) for l in layers)
        print(f"Mạng: {chuoi}")
        print(f"batch = {X.shape[0]}, dtype = {X.dtype}, loss = {loss:.6f}\n")
        print(f"{'tham số':<10}{'shape':<12}{'sai số tương đối':<20}kết quả")
        print("-" * 58)

    for ten, g_analytic in giai_tich.items():
        g_numeric = numerical_gradient(f, tham_so[ten])
        sai_so = rel_error(g_analytic, g_numeric)
        dat = sai_so < nguong
        if verbose:
            print(f"{ten:<10}{str(g_analytic.shape):<12}{sai_so:<20.3e}{'ĐẠT' if dat else 'SAI'}")
        assert dat, (
            f"{ten}: sai số {sai_so:.3e} >= {nguong:.0e}. Xem lại công thức backward.\n"
            f"giải tích:\n{g_analytic}\nsố:\n{g_numeric}"
        )

    if verbose:
        print("\nTẤT CẢ GRADIENT ĐỀU ĐÚNG ✔")


def check_gradients_chong_overfit(seed: int = 0, nguong: float = 1e-6) -> None:
    """
    Gradcheck cho mạng có BatchNorm và Dropout:
        Dense(5,4) -> BatchNorm1d(4) -> ReLU -> Dropout(0.3, seed cố định) -> Dense(4,3)
    """
    from layers import BatchNorm1d, Dropout

    rng = np.random.default_rng(seed)
    B, D, H, C = 7, 5, 4, 3

    X = rng.standard_normal((B, D))
    Y = np.eye(C)[rng.integers(0, C, B)]

    layers = [
        Dense(D, H, init="he", rng=rng, dtype=np.float64),
        BatchNorm1d(H, dtype=np.float64),
        ReLU(),
        Dropout(0.3, seed=123),          # seed cố định -> mask giống nhau mọi lần gọi
        Dense(H, C, init="xavier", rng=rng, dtype=np.float64),
    ]
    check_gradients_mang(layers, X, Y, nguong=nguong)


def overfit_batch_nho(n_anh: int = 64, so_vong: int = 300, lr: float = 0.1,
                      seed: int = 42) -> None:
    """
    Học thuộc lòng một batch nhỏ ảnh THẬT bằng SGD viết tay.

    Parameters:
        n_anh: Số ảnh lấy từ tập train.
        so_vong: Số vòng lặp cập nhật.
        lr: Learning rate.
        seed: Hạt giống ngẫu nhiên.

    Đây là bài test chuẩn của giới ML ("overfit a tiny batch"). Mạng học THUỘC
    được vài chục ảnh là điều MONG MUỐN ở đây: nếu không thuộc nổi thì chắc chắn
    gradient đang sai, khỏi cần tìm lỗi ở learning rate hay kiến trúc.
    """
    from data_loader import CLASS_NAMES
    from losses import accuracy
    from preprocess import prepare_data

    data = prepare_data()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["X_train"]), n_anh, replace=False)
    X, Y, y = data["X_train"][idx], data["Y_train"][idx], data["y_train"][idx]

    dense1 = Dense(X.shape[1], 128, init="he", rng=rng)
    relu = ReLU()
    dense2 = Dense(128, len(CLASS_NAMES), init="xavier", rng=rng)
    criterion = SoftmaxCrossEntropy()

    print(f"Học thuộc {n_anh} ảnh thật | lr = {lr} | {so_vong} vòng")
    print(f"{'vòng':<8}{'loss':<12}accuracy")
    print("-" * 32)

    for vong in range(so_vong + 1):
        # 1. Chạy tiến
        Z = dense2.forward(relu.forward(dense1.forward(X)))
        loss = criterion.forward(Z, Y)

        if vong % (so_vong // 6) == 0:
            print(f"{vong:<8}{loss:<12.4f}{accuracy(Z, y):.4f}")

        # 2. Lan ngược
        dense1.backward(relu.backward(dense2.backward(criterion.backward())))

        # 3. Cập nhật tham số: SGD viết tay (Bước 7 sẽ đóng gói thành class)
        for lop in (dense1, dense2):
            lop.W -= lr * lop.dW
            lop.b -= lr * lop.db

    Z = dense2.forward(relu.forward(dense1.forward(X)))
    loss_cuoi, acc_cuoi = criterion.forward(Z, Y), accuracy(Z, y)
    print(f"\nKết quả: loss = {loss_cuoi:.4f}, accuracy = {acc_cuoi:.4f}")
    assert loss_cuoi < 0.1 and acc_cuoi == 1.0, (
        "Không học thuộc nổi batch nhỏ -> gradient nhiều khả năng đang SAI"
    )
    print("Mạng học thuộc được batch nhỏ -> backprop hoạt động đúng ✔")


if __name__ == "__main__":
    print("=" * 56)
    print("PHẦN 1: KIỂM TRA GRADIENT BẰNG SAI PHÂN SỐ")
    print("=" * 56)
    check_gradients()

    print("\n" + "=" * 56)
    print("PHẦN 1b: GRADCHECK CHO BATCHNORM VÀ DROPOUT")
    print("=" * 56)
    check_gradients_chong_overfit()

    print("\n" + "=" * 56)
    print("PHẦN 2: HỌC THUỘC MỘT BATCH NHỎ (dữ liệu NEU-DET thật)")
    print("=" * 56)
    overfit_batch_nho()
