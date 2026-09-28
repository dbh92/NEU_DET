import numpy as np
from layers import Layer

class Optimizer:
    """
    Lớp cơ sở: gom tham chiếu tới toàn bộ tham số và gradient của mạng.
    """

    def __init__(self, layers: list[Layer], lr: float = 0.01,
                 weight_decay: float = 0.0):
        """
        Parameters:
            layers: Danh sách các lớp của mạng. Lớp không có tham số (ReLU, Dropout ...) trả về {} nên tự động bị bỏ qua
            lr: Learning rate.
            weight_decay: Hệ số L2. Chỉ áp lên trọng số "W", KHÔNG áp lên bias
                hay gamma/beta của BatchNorm — xem self.la_trong_so.
        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.params: list[np.ndarray] = []       # [W1, b1, W2, b2, ...]
        self.grads: list[np.ndarray] = []        # [dW1, db1, dW2, db2, ...] CÙNG THỨ TỰ
        self.la_trong_so: list[bool] = []        # True nếu tham số đó là "W"

        # Duyệt MỘT LẦN ở đây, không gọi lại params() trong step(), để thứ tự
        # của hai danh sách luôn khớp nhau: params[i] luôn ứng với grads[i].
        for layer in layers:
            p, g = layer.params(), layer.grads()
            for ten in p:
                if ten not in g:
                    raise KeyError(f"{type(layer).__name__}: có params['{ten}'] nhưng thiếu grads['{ten}']")
                if p[ten].shape != g[ten].shape:
                    raise ValueError(
                        f"{type(layer).__name__}.{ten}: params shape {p[ten].shape} "
                        f"khác grads shape {g[ten].shape}"
                    )
                self.params.append(p[ten])
                self.grads.append(g[ten])
                # Chỉ W mới bị phạt L2: b chỉ dịch ngưỡng kích hoạt, không nhân
                # với dữ liệu nên không gây overfit; gamma/beta của BatchNorm cũng vậy.
                self.la_trong_so.append(ten == "W")

        if not self.params:
            raise ValueError("Không tìm thấy tham số nào — mạng chỉ gồm các lớp không có tham số?")
        if len(self.params) != len(self.grads):
            raise RuntimeError(f"params ({len(self.params)}) và grads ({len(self.grads)}) lệch nhau")

    def step(self) -> None:
        """Cập nhật toàn bộ tham số bằng gradient hiện tại."""
        raise NotImplementedError("Lớp con phải tự cài đặt step()")

    def grad_hieu_dung(self, i: int) -> np.ndarray:
        """
        Gradient của tham số thứ i, đã cộng số hạng phạt L2.

        L_tổng = L_CE + (lambda/2) * sum(W^2)  =>  dL/dW = dL_CE/dW + lambda*W
        """
        if self.weight_decay > 0.0 and self.la_trong_so[i]:
            return self.grads[i] + self.weight_decay * self.params[i]
        return self.grads[i]

    def l2_loss(self) -> float:
        """Giá trị số hạng phạt (lambda/2) * sum(W^2), để cộng vào loss khi in ra."""
        if self.weight_decay == 0.0:
            return 0.0
        tong = sum(float((p ** 2).sum())
                   for p, la_w in zip(self.params, self.la_trong_so) if la_w)
        return 0.5 * self.weight_decay * tong

    def zero_grad(self) -> None:
        """
        Xoá gradient về 0.

        Hiện CHƯA bắt buộc, vì Dense.backward ghi đè hoàn toàn (self.dW[...] = ...)
        chứ không cộng dồn. Nhưng ở Bước 10, gradient của L2 sẽ được CỘNG vào dW,
        lúc đó phải xoá trước mỗi vòng.
        """
        for g in self.grads:
            g[...] = 0

    def so_tham_so(self) -> int:
        """Tổng số tham số học được mà optimizer đang quản lý."""
        return sum(p.size for p in self.params)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lr={self})"


class SGD (Optimizer):
    """
    Stochastic Gradient Descent: w <- w - lr * g

    Luôn đi theo hướng dốc nhất TẠI ĐIỂM HIỆN TẠI, không nhớ gì về quá khứ.
    Trong "thung lũng hẹp" (hướng này dốc, hướng kia thoai thoải) nó nảy qua
    nảy lại giữa hai vách và tiến rất chậm về đáy.
    """

    def step(self):
        for i, p in enumerate(self.params):
            # p -= ... là GHI TẠI CHỖ (__isub__) nên layer.W thật sự đổi.
            # Viết p = p - lr*g thì chỉ gán lại biến local p, layer.W KHÔNG đổi
            # -> mạng chạy bình thường, loss đứng yên, và không có lỗi nào được báo.
            p -= self.lr * self.grad_hieu_dung(i)


class Momentum(Optimizer):
    """
    SGD có quán tính:  v <- mu*v - lr*g ;  w <- w + v

    Như hòn bi lăn xuống dốc. Dao động qua lại triệt tiêu lẫn nhau, còn hướng
    đi đều đặn thì được cộng dồn.

    mu = 0.9 nghĩa là giữ lại 90% vận tốc cũ, nên bước đi lúc ổn định lớn gấp
    khoảng 1/(1-0.9) = 10 lần SGD cùng lr. => Chuyển từ SGD sang Momentum thì
    nên GIẢM lr khoảng 10 lần, nếu không mạng sẽ nổ.
    """

    def __init__(self, layers: list[Layer], lr: float = 0.01, momentum: float = 0.9):
        super().__init__(layers, lr)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum phải nằm trong [0, 1), nhận {momentum}")
        self.momentum = momentum
        # Trạng thái: vận tốc, cùng shape và dtype với từng tham số
        self.velocities = [np.zeros_like(p) for p in self.params]

    def step(self) -> None:
        for i, (p, v) in enumerate(zip(self.params, self.velocities)):
            v *= self.momentum                          # giữ lại quán tính cũ
            v -= self.lr * self.grad_hieu_dung(i)       # cộng thêm lực đẩy mới
            p += v

    def __repr__(self) -> str:
        return f"Momentum(lr={self.lr}, momentum={self.momentum})"


class Adam(Optimizer):
    """
    Adam: mỗi tham số một learning rate riêng, tự điều chỉnh.

        m <- b1*m + (1-b1)*g            trung bình trượt của gradient  (momentum)
        v <- b2*v + (1-b2)*g^2          trung bình trượt của BÌNH PHƯƠNG gradient
        m_hat = m / (1 - b1^t)          hiệu chỉnh chệch
        v_hat = v / (1 - b2^t)
        w <- w - lr * m_hat / (sqrt(v_hat) + eps)

    Chia cho sqrt(v_hat) làm bước đi KHÔNG phụ thuộc độ lớn gradient: tham số
    có gradient bé tí vẫn đi được bước ra trò. Vì m_hat/sqrt(v_hat) luôn cỡ 1,
    bước đi thực tế luôn xấp xỉ lr -> lr của Adam nhỏ hơn SGD rất nhiều (1e-3).

    Hiệu chỉnh chệch: m và v khởi tạo bằng 0 nên những bước đầu bị kéo lệch về 0.
    Ở t = 1, m = 0.1*g (nhỏ hơn thật 10 lần), chia cho (1 - 0.9^1) = 0.1 thì
    khôi phục đúng g. Khi t lớn, b1^t -> 0 nên hiệu chỉnh tự biến mất.
    """

    def __init__(self, layers: list[Layer], lr:float = 1e-3, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999,
                 eps: float = 1e-8, 
                 bias_correction: bool = True,
                 weight_decay: float = 0.0,
                 decoupled: bool = True):
        """
        Parameters:
            bias_correction: Đặt False để THẤY tác hại của việc quên hiệu chỉnh
                chệch (chỉ dùng để so sánh trong test, đừng dùng khi train thật).
            weight_decay: Hệ số L2, chỉ áp lên "W".
            decoupled: True = AdamW. Trừ thẳng lr*wd*p SAU bước Adam, thay vì
                cộng lambda*W vào gradient rồi để nó bị chia cho sqrt(v_hat).
                Cách cộng vào gradient (decoupled=False) làm mức phạt bị bóp méo:
                trọng số nào có gradient lớn thì v_hat lớn, nên bị phạt NHẸ đi —
                trong khi lẽ ra mọi trọng số phải bị co về 0 với cùng một tỉ lệ.
        """
        super().__init__(layers, lr, weight_decay=0.0 if decoupled else weight_decay)
        self.decoupled = decoupled
        self.wd_tach_roi = weight_decay if decoupled else 0.0
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta phải nằm trong [0, 1), nhận beta1={beta1}, beta2={beta2}")
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.bias_correction = bias_correction

        self.m = [np.zeros_like(p) for p in self.params]   # moment bậc 1 m (Moment 1 - Trung bình): Lưu lại hướng đi (đà quán tính).
        self.v = [np.zeros_like(p) for p in self.params]   # moment bậc 2 v (Moment 2 - Phương sai): Lưu lại "độ lớn, độ rung lắc" của gradient.
        self.t = 0     

    def step(self) -> None:
        self.t += 1     # MỘT lần cho cả vòng lặp, không phải mỗi tham số

        # Hệ số hiệu chỉnh chung cho mọi tham số ở bước t này
        if self.bias_correction:
            hc1 = 1.0 - self.beta1 ** self.t
            hc2 = 1.0 - self.beta2 ** self.t
        else:
            hc1 = hc2 = 1.0

        for i, (p, m, v) in enumerate(zip(self.params, self.m, self.v)):
            g = self.grad_hieu_dung(i)      # đã cộng L2 nếu decoupled=False

            # m += (1-b1)*(g - m)  tương đương  m = b1*m + (1-b1)*g, nhưng ghi tại chỗ
            m += (1.0 - self.beta1) * (g - m)
            v += (1.0 - self.beta2) * (g * g - v)

            m_hat = m / hc1
            v_hat = v / hc2

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # AdamW: co trọng số về 0 bằng một bước riêng, không đi qua sqrt(v_hat)
            if self.wd_tach_roi > 0.0 and self.la_trong_so[i]:
                p -= self.lr * self.wd_tach_roi * p

    def l2_loss(self) -> float:
        """Với AdamW không có số hạng L2 trong loss, nhưng vẫn báo cáo để so sánh."""
        wd = self.wd_tach_roi if self.decoupled else self.weight_decay
        if wd == 0.0:
            return 0.0
        tong = sum(float((p ** 2).sum())
                   for p, la_w in zip(self.params, self.la_trong_so) if la_w)
        return 0.5 * wd * tong

    def __repr__(self) -> str:
        ten = "Adam" if self.bias_correction else "Adam(KHÔNG hiệu chỉnh chệch)"
        if self.decoupled and self.wd_tach_roi > 0:
            ten = "AdamW"
        wd = self.wd_tach_roi if self.decoupled else self.weight_decay
        them = f", weight_decay={wd}" if wd else ""
        return f"{ten}(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}{them})"


def test_optimizers(seed: int = 42) -> None:
    """Kiểm thử: bẫy ghi tại chỗ, thung lũng hẹp, hiệu chỉnh chệch, mạng thật, dtype."""
    from activations import ReLU
    from data_loader import CLASS_NAMES
    from layers import Dense
    from losses import SoftmaxCrossEntropy, accuracy
    from preprocess import prepare_data

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST 3: Mạng thật — học thuộc 64 ảnh NEU-DET (cùng seed, cùng W ban đầu)")
    print("=" * 72)
    data = prepare_data()
    idx = np.random.default_rng(seed).choice(len(data["X_train"]), 64, replace=False)
    X, Y, y = data["X_train"][idx], data["Y_train"][idx], data["y_train"][idx]

    def tao_mang():
        """Dựng lại mạng với ĐÚNG bộ trọng số ban đầu, để so sánh công bằng."""
        rng = np.random.default_rng(seed)
        d1 = Dense(X.shape[1], 128, init="he", rng=rng)
        relu = ReLU()
        d2 = Dense(128, len(CLASS_NAMES), init="xavier", rng=rng)
        return [d1, relu, d2], d1, relu, d2

    moc = [0, 25, 50, 100, 200]
    print(f"{'thuật toán':<26}" + "".join(f"vòng {m:<8}" for m in moc)
          + f"{'đỉnh loss':<12}accuracy")
    print("-" * 94)

    for ten, tao_opt in [
        ("SGD(lr=0.1)", lambda ls: SGD(ls, lr=0.1)),
        ("Momentum(lr=0.01)", lambda ls: Momentum(ls, lr=0.01, momentum=0.9)),
        ("Adam(lr=1e-3)", lambda ls: Adam(ls, lr=1e-3)),
        ("Adam(1e-3) KHÔNG hc chệch", lambda ls: Adam(ls, lr=1e-3, bias_correction=False)),
    ]:
        layers, d1, relu, d2 = tao_mang()
        crit = SoftmaxCrossEntropy()
        opt = tao_opt(layers)
        lich_su, dinh = {}, 0.0

        for vong in range(max(moc) + 1):
            Z = d2.forward(relu.forward(d1.forward(X)))
            loss = crit.forward(Z, Y)
            dinh = max(dinh, loss)
            if vong in moc:
                lich_su[vong] = loss
            d1.backward(relu.backward(d2.backward(crit.backward())))
            opt.step()

        Z = d2.forward(relu.forward(d1.forward(X)))
        print(f"{ten:<26}" + "".join(f"{lich_su[m]:<13.4f}" for m in moc)
              + f"{dinh:<12.2f}{accuracy(Z, y):.4f}")
        assert d1.W.dtype == np.float32, "dtype bị đẩy lên float64!"

    print("\n-> Cột 'đỉnh loss' cho thấy tác hại thật của việc bỏ hiệu chỉnh chệch:")
    print("   loss vọt lên cao hơn hẳn ở những vòng đầu vì bước đi quá đà.")
    print("   Trên bài toán trơn tru thì vẫn về đích, nhưng với dữ liệu nhiễu")
    print("   hoặc lr lớn hơn thì đó là lúc mạng nổ.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST 4: dtype giữ nguyên float32 sau nhiều bước (kiểm tra kỹ với Adam)")
    print("=" * 72)
    lop = Dense(4, 3, rng=np.random.default_rng(seed))
    opt = Adam([lop], lr=1e-3)
    for _ in range(50):
        lop.dW[...] = 0.01
        lop.db[...] = 0.01
        opt.step()
    print(f"W {lop.W.dtype} | b {lop.b.dtype} | m {opt.m[0].dtype} | v {opt.v[0].dtype} | t = {opt.t}")
    assert lop.W.dtype == lop.b.dtype == opt.m[0].dtype == opt.v[0].dtype == np.float32
    assert opt.t == 50, "t phải tăng 1 lần mỗi step(), không phải mỗi tham số"

    print("\n" + "=" * 72)
    print("TẤT CẢ TEST ĐỀU ĐẠT ✔")
    print("=" * 72)


if __name__ == "__main__":
    test_optimizers()
                                   # số bước đã đi