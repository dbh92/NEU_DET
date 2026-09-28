"""
Dự đoán loại lỗi bề mặt cho một hoặc nhiều ảnh, từ dòng lệnh.

    python src/predict.py <đường dẫn ảnh> [thêm ảnh...]
    python src/predict.py                     # không tham số: lấy vài ảnh val làm ví dụ

Cần có checkpoints/cnn_neu_det.npz (tạo bằng: python -u src/train_final.py).
"""

import os
import sys

import numpy as np

from checkpoint import du_doan_nhieu_anh, nap_model
from data_loader import CLASS_NAMES, DATA_ROOT
from train_final import DUONG_DAN_CHECKPOINT


def anh_vi_du(so_luong_moi_lop: int = 1, seed: int = 0) -> list[str]:
    """Lấy vài ảnh ngẫu nhiên từ tập validation để chạy thử."""
    rng = np.random.default_rng(seed)
    duong_dan = []
    for ten_lop in CLASS_NAMES:
        thu_muc = os.path.join(DATA_ROOT, "validation", "images", ten_lop)
        for f in rng.choice(sorted(os.listdir(thu_muc)), so_luong_moi_lop, replace=False):
            duong_dan.append(os.path.join(thu_muc, f))
    return duong_dan


def main(argv: list[str]) -> int:
    cac_anh = argv[1:]
    tu_vi_du = not cac_anh
    if tu_vi_du:
        cac_anh = anh_vi_du()
        print("Không có tham số -> dùng ảnh ví dụ từ tập validation.\n")

    thieu = [p for p in cac_anh if not os.path.exists(p)]
    if thieu:
        print("Không tìm thấy các file sau:", *thieu, sep="\n  ")
        return 1

    try:
        model, meta = nap_model(DUONG_DAN_CHECKPOINT, verbose=False)
    except FileNotFoundError as e:
        print(e)
        return 1

    ten_lop = meta["class_names"]
    print(f"Mô hình: {meta.get('mo_ta', '?')}")
    print(f"val accuracy {meta.get('val_acc', float('nan')):.4f} | "
          f"tạo ngày {meta.get('ngay_tao', '?')}\n")

    P = du_doan_nhieu_anh(model, meta, cac_anh)

    dung = 0
    for duong_dan, xac_suat in zip(cac_anh, P):
        ten_file = os.path.basename(duong_dan)
        thu_tu = np.argsort(-xac_suat)[:3]
        print(f"Ảnh: {ten_file}")
        for hang, c in enumerate(thu_tu):
            dau = "->" if hang == 0 else "  "
            print(f"  {dau} {ten_lop[c]:<18}{xac_suat[c]:.3f}")

        if tu_vi_du:
            # Ảnh ví dụ nằm trong thư mục mang tên lớp, nên đối chiếu được
            that = os.path.basename(os.path.dirname(duong_dan))
            doan = ten_lop[int(thu_tu[0])]
            dung += doan == that
            print(f"     nhãn thật: {that}  {'✔' if doan == that else '✘'}")
        print()

    if tu_vi_du:
        print(f"Đúng {dung}/{len(cac_anh)} ảnh ví dụ.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
