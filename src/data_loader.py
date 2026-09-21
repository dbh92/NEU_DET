import numpy as np
from tqdm import tqdm
import cv2
import os


CLASS_NAMES = [
    "crazing", 
    "inclusion", 
    "patches",
    "pitted_surface", 
    "rolled-in_scale", 
    "scratches"
]

IMG_SIZE = 64   # kích thước ảnh đầu ra (gốc 200 x 200 -> resize về 64x64)

# Đường dẫn tính theo vị trí file này (src/data_loader.py), KHÔNG theo thư mục đang đứng (cwd),
# nên chạy script từ thư mục nào cũng trỏ đúng một chỗ.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "NEU-DET")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache", f"neu_det_{IMG_SIZE}.npz")


def resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resize ảnh (2D array) về kích thước (size, size) sử dụng thuật toán nearest-neighbor.
    Áp dụng vector hóa (fancy indexing) để tối ưu tốc độ, không dùng vòng lặp for.
    
    Args:
        image (np.ndarray): Mảng 2 chiều (H, W), dtype uint8.
        size (int): Chiều dài cạnh của ảnh đầu ra (ví dụ: IMG_SIZE).
        
    Returns:
        np.ndarray: Mảng 2 chiều kích thước (size, size), dtype uint8.
    """
    H, W = image.shape

    # 1. Tạo mảng chỉ số hàng nguồn: shape (size,)
    row_idx = np.arange(size) * H // size  # scale down to original height

    # 2. Tạo mảng chỉ số cột nguồn: shape (size,)
    col_idx = np.arange(size) * W // size  # scale down to original width

    # 3. Lấy pixel từ ảnh gốc bằng fancy indexing
    resized_image = image[row_idx][:, col_idx] # shape (size, size)

    return resized_image


def load_image(path: str, size: int = IMG_SIZE) -> np.ndarray:
    """
    Load ảnh từ đường dẫn và resize về kích thước (size, size).
    
    Args:
        path (str): Đường dẫn tới file ảnh.
        size (int): Kích thước mong muốn của ảnh đầu ra.
        
    Returns:
        np.ndarray: Mảng 2 chiều kích thước (size, size), dtype uint8.
    """
    # Load ảnh bằng OpenCV
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # đọc ảnh xám
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại {path}")
    
    # Resize ảnh
    resized_img = resize_nearest(img, size)
    
    return resized_img


def load_split(split_dir: str, size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    Load tất cả ảnh trong thư mục split (train/val/test) và nhãn tương ứng.
    
    Args:
        split_dir (str): Đường dẫn tới thư mục split (ví dụ: 'data/train').
        size (int): Kích thước mong muốn của ảnh đầu ra.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - X: Mảng 3 chiều (N, size, size), dtype uint8.
            - y: Mảng 1 chiều (N,), dtype int (chỉ số lớp).
    """
    images_dir = os.path.join(split_dir, "images")
    X_list = []
    y_list = []
    for label, name in enumerate(CLASS_NAMES):
        folder = os.path.join(images_dir, name)
        files = sorted(os.listdir(folder))
        for fname in tqdm(files, desc=f"Loading images from {name}"):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            X_list.append(load_image(os.path.join(folder, fname), size))
            y_list.append(label)
    X = np.stack(X_list)            # list of (64,64) -> (N,64,64)
    y = np.array(y_list, dtype=np.int64)

    return X, y

def build_cache(root: str = DATA_ROOT,
                cache_path: str = CACHE_PATH,
                size: int = IMG_SIZE) -> None:
    """
    Đọc toàn bộ ảnh train + validation và lưu vào 1 file .npz nén.

    Args:
        root (str): Thư mục gốc của dataset (chứa train/ và validation/).
        cache_path (str): Đường dẫn file .npz đầu ra.
        size (int): Kích thước ảnh sau resize.
    """
    X_train, y_train = load_split(os.path.join(root, "train"), size)
    X_val, y_val = load_split(os.path.join(root, "validation"), size)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val)

    print(f"Đã lưu cache: {cache_path}")
    for name, arr in [("X_train", X_train), ("y_train", y_train),
                      ("X_val", X_val), ("y_val", y_val)]:
        print(f"  {name:8s} {str(arr.shape):18s} {arr.dtype}")


def load_cache(cache_path: str = CACHE_PATH
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Nạp dữ liệu từ file cache .npz. Nếu chưa có cache thì build trước.

    Args:
        cache_path (str): Đường dẫn file .npz.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
            - X_*: (N, size, size) uint8
            - y_*: (N,) int64
    """
    if not os.path.exists(cache_path):
        print(f"Chưa có cache, đang build: {cache_path}")
        build_cache(cache_path=cache_path)

    with np.load(cache_path) as data:
        return data["X_train"], data["y_train"], data["X_val"], data["y_val"]


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_cache()
    print(X_train.shape, X_train.dtype, X_train.min(), X_train.max())
    print(y_train.shape, np.bincount(y_train))
    print(X_val.shape, np.bincount(y_val))
