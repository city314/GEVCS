import numpy as np
import cv2


def floyd_steinberg_halftone(img: np.ndarray) -> np.ndarray:
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

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def floyd_steinberg_constrained_secret(secret, n1, n2, m=16):
    # secret cần về dạng [0..1]
    s = secret.astype(np.float32) / 255.0
    h, w = s.shape

    err = np.zeros_like(s, dtype=np.float32)
    nS = np.zeros_like(secret, dtype=np.int32)

    for y in range(h):
        for x in range(w):
            val = s[y, x] + err[y, x]
            val = np.clip(val, 0, 1)
            ideal = int(round(val * m))

            n1_xy = int(n1[y, x])
            n2_xy = int(n2[y, x])
            low = max(0, n1_xy + n2_xy - m)
            high = min(n1_xy, n2_xy)

            actual = int(np.clip(ideal, low, high))
            nS[y, x] = actual

            val_hat = actual / m
            diff = val - val_hat

            if x + 1 < w:
                err[y, x + 1] += diff * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    err[y + 1, x - 1] += diff * 3 / 16
                err[y + 1, x] += diff * 5 / 16
                if x + 1 < w:
                    err[y + 1, x + 1] += diff * 1 / 16

    return nS


def compress_dynamic_range_host(img, L=0.55):
    f = img.astype(np.float32) / 255.0
    f = (f - f.min()) / (f.max() - f.min() + 1e-6)
    f = L + (1 - L) * f
    return (f * 255.0).astype(np.uint8)


def compress_dynamic_range_secret(img):
    f = img.astype(np.float32) / 255.0
    f = (f - f.min()) / (f.max() - f.min() + 1e-6)
    return (f * 255.0).astype(np.uint8)


def gevcs_generate_shares_ross(host1_gray, host2_gray, secret_gray, m=16, L=0.55, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    H, W = secret_gray.shape
    n = int(np.sqrt(m))

    # Normalize host
    f1 = host1_gray.astype(np.float32) / 255.0
    f1 = L + (1 - L) * (f1 - f1.min()) / (f1.max() - f1.min() + 1e-6)

    f2 = host2_gray.astype(np.float32) / 255.0
    f2 = L + (1 - L) * (f2 - f2.min()) / (f2.max() - f2.min() + 1e-6)

    n1 = np.rint(f1 * m).astype(np.int32)
    n2 = np.rint(f2 * m).astype(np.int32)

    # Secret normalization (fix quan trọng)
    secret_scaled = compress_dynamic_range_secret(secret_gray)
    nS = floyd_steinberg_constrained_secret(secret_scaled, n1, n2, m=m)

    share1 = np.zeros((H * n, W * n), dtype=np.uint8)
    share2 = np.zeros((H * n, W * n), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            A = nT = int(np.clip(nS[y, x], max(0, n1[y, x] + n2[y, x] - m), min(n1[y, x], n2[y, x])))
            B = n1[y, x] - nT
            C = n2[y, x] - nT
            D = m - (A + B + C)

            idx = np.arange(m)
            rng.shuffle(idx)

            blk1 = np.zeros(m, np.uint8)
            blk2 = np.zeros(m, np.uint8)

            p = 0
            for k in idx[p:p+A]:
                blk1[k] = blk2[k] = 255
            p += A
            for k in idx[p:p+B]:
                blk1[k] = 255
            p += B
            for k in idx[p:p+C]:
                blk2[k] = 255

            blk1 = blk1.reshape(n, n)
            blk2 = blk2.reshape(n, n)

            share1[y*n:(y+1)*n, x*n:(x+1)*n] = blk1
            share2[y*n:(y+1)*n, x*n:(x+1)*n] = blk2

    return share1, share2


def overlay_shares(s1, s2):
    # Reconstruction đúng kiểu Ross: AND logic → minimum
    return np.minimum(s1, s2)


def enhance_reconstruction(img):
    # 1. Normalize mạnh
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # 2. CLAHE mạnh hơn
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    img = clahe.apply(img)

    # 3. Sharpen mạnh
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharp = cv2.addWeighted(img, 2.0, blur, -1.0, 0)

    return sharp.astype(np.uint8)

