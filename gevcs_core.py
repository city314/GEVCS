import numpy as np
import cv2


def load_and_prepare_images(path_host1, path_host2, path_secret, size=(256, 256)):
    """
    load 3 ảnh, convert sang grayscale, resize cùng size.
    trả về: host1, host2, secret (uint8, [0,255])
    """
    def _load(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"cannot load image from {path}")
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    host1 = _load(path_host1)
    host2 = _load(path_host2)
    secret = _load(path_secret)
    return host1, host2, secret


def floyd_steinberg_halftone(img):
    """
    halftone đơn giản (Floyd–Steinberg).
    input: ảnh xám uint8
    output: ảnh nhị phân (0 hoặc 255)
    """
    img = img.astype(np.float32) / 255.0
    h, w = img.shape
    out = img.copy()

    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                out[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0:
                    out[y + 1, x - 1] += err * 3 / 16
                out[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    out[y + 1, x + 1] += err * 1 / 16

    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def gevcs_generate_shares(host1, host2, secret, m=9, rng=None):
    """
    GEVCS grayscale đơn giản (2-out-of-2).
    - host1, host2, secret: ảnh xám uint8, cùng shape
    - m: pixel expansion, ở đây chỉ support m=9 (block 3x3)
    trả về: share1, share2 (0/255)
    """
    if rng is None:
        rng = np.random.default_rng()

    # m phải là bình phương nguyên (2x2, 3x3, 4x4, ...)
    n = int(np.sqrt(m))
    if n * n != m:
        raise ValueError("m must be a perfect square, e.g. 4, 9, 16")

    if host1.shape != host2.shape or host1.shape != secret.shape:
        raise ValueError("all images must have the same shape")

    h, w = secret.shape
    H = h * n
    W = w * n

    share1 = np.ones((H, W), dtype=np.uint8) * 255
    share2 = np.ones((H, W), dtype=np.uint8) * 255

    # tạo list index trong block n×n
    block_indices = [(by, bx) for by in range(n) for bx in range(n)]

    host1_f = host1.astype(np.float32) / 255.0
    host2_f = host2.astype(np.float32) / 255.0
    secret_f = secret.astype(np.float32) / 255.0

    for y in range(h):
        for x in range(w):
            d_host1 = 1.0 - host1_f[y, x]
            d_host2 = 1.0 - host2_f[y, x]
            d_secret = 1.0 - secret_f[y, x]

            nb_secret = int(round(0.7 * (d_secret ** 0.8) * m))
            nb_host1 = int(round(d_host1 * m))
            nb_host2 = int(round(d_host2 * m))

            nb_secret = max(0, min(m, nb_secret))
            nb_host1 = max(0, min(m, nb_host1))
            nb_host2 = max(0, min(m, nb_host2))

            idxs = np.arange(m)
            rng.shuffle(idxs)
            overlap_idxs = idxs[:nb_secret]
            remaining = idxs[nb_secret:]

            extra1 = max(0, nb_host1 - nb_secret)
            extra2 = max(0, nb_host2 - nb_secret)

            rng.shuffle(remaining)
            share1_only = remaining[:extra1]
            share2_only = remaining[extra1:extra1 + extra2]

            for k in range(m):
                by, bx = block_indices[k]
                sy = y * n + by
                sx = x * n + bx
                if k in overlap_idxs:
                    share1[sy, sx] = 0
                    share2[sy, sx] = 0
                elif k in share1_only:
                    share1[sy, sx] = 0
                elif k in share2_only:
                    share2[sy, sx] = 0

    return share1, share2

def gevcs_generate_shares_2of3(host1, host2, host3, secret, m=9, rng=None):
    """
    GEVCS grayscale demo cho scheme (2,3):
    - input: 3 ảnh host + 1 ảnh secret (đã halftone, cùng shape)
    - output: 3 share (0/255). Bất kỳ 2 share chồng lên sẽ thấy secret (do
      tồn tại vùng subpixel đen chung cho cả 3 share).
    NOTE: đây là bản demo, không phải scheme tối ưu / chứng minh bảo mật.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = int(np.sqrt(m))
    if n * n != m:
        raise ValueError("m must be a perfect square, e.g. 4, 9, 16")

    if not (host1.shape == host2.shape == host3.shape == secret.shape):
        raise ValueError("all images must have the same shape")

    h, w = secret.shape
    H = h * n
    W = w * n

    share1 = np.ones((H, W), dtype=np.uint8) * 255
    share2 = np.ones((H, W), dtype=np.uint8) * 255
    share3 = np.ones((H, W), dtype=np.uint8) * 255

    block_indices = [(by, bx) for by in range(n) for bx in range(n)]

    host1_f = host1.astype(np.float32) / 255.0
    host2_f = host2.astype(np.float32) / 255.0
    host3_f = host3.astype(np.float32) / 255.0
    secret_f = secret.astype(np.float32) / 255.0

    for y in range(h):
        for x in range(w):
            d_host1 = 1.0 - host1_f[y, x]
            d_host2 = 1.0 - host2_f[y, x]
            d_host3 = 1.0 - host3_f[y, x]
            d_secret = 1.0 - secret_f[y, x]

            # số subpixel đen cho secret + cho từng host
            # dùng cùng “style” với (2,2): hệ số 0.7 để secret không quá đậm
            nb_secret = int(round(0.7 * (d_secret ** 0.8) * m))
            nb_host1 = int(round(d_host1 * m))
            nb_host2 = int(round(d_host2 * m))
            nb_host3 = int(round(d_host3 * m))

            nb_secret = max(0, min(m, nb_secret))
            nb_host1 = max(0, min(m, nb_host1))
            nb_host2 = max(0, min(m, nb_host2))
            nb_host3 = max(0, min(m, nb_host3))

            idxs = np.arange(m)
            rng.shuffle(idxs)

            # vị trí overlap chung cho cả 3 share → đảm bảo bất kỳ 2 share cũng thấy secret
            all_idxs = idxs[:nb_secret]
            remaining = idxs[nb_secret:]

            extra1 = max(0, nb_host1 - nb_secret)
            extra2 = max(0, nb_host2 - nb_secret)
            extra3 = max(0, nb_host3 - nb_secret)

            # lấy thêm index cho từng share (có thể trùng nhau → overlap theo cặp)
            if len(remaining) == 0:
                rem = np.array([0], dtype=int)
            else:
                rem = remaining

            s1_extra = rng.choice(rem, size=extra1, replace=True) if extra1 > 0 else np.array([], dtype=int)
            s2_extra = rng.choice(rem, size=extra2, replace=True) if extra2 > 0 else np.array([], dtype=int)
            s3_extra = rng.choice(rem, size=extra3, replace=True) if extra3 > 0 else np.array([], dtype=int)

            s1_black = set(all_idxs.tolist() + s1_extra.tolist())
            s2_black = set(all_idxs.tolist() + s2_extra.tolist())
            s3_black = set(all_idxs.tolist() + s3_extra.tolist())

            for k in range(m):
                by, bx = block_indices[k]
                sy = y * n + by
                sx = x * n + bx
                if k in s1_black:
                    share1[sy, sx] = 0
                if k in s2_black:
                    share2[sy, sx] = 0
                if k in s3_black:
                    share3[sy, sx] = 0

    return share1, share2, share3

def overlay_shares(share1, share2, op="and"):
    """
    chồng 2 share.
    - op="and": kiểu visual crypto classic (đen nếu có share nào đen)
    - op="or": trắng nếu có share nào trắng
    trả về: ảnh nhị phân 0/255
    """
    if share1.shape != share2.shape:
        raise ValueError("share1 and share2 must have same shape")

    s1 = (share1 == 0).astype(np.uint8)
    s2 = (share2 == 0).astype(np.uint8)

    if op == "and":
        combined = np.clip(s1 + s2, 0, 1)  # 0 hoặc 1
    elif op == "or":
        combined = np.logical_and(s1, s2).astype(np.uint8)
    else:
        raise ValueError("op must be 'and' or 'or'")

    # 1 -> đen, 0 -> trắng
    return (1 - combined) * 255
