import os
import cv2
import numpy as np
from typing import List, Tuple

from gevcs_core import (
    floyd_steinberg_halftone,
    gevcs_generate_shares,
    overlay_shares,
    gevcs_generate_shares_2of3
)

def center_crop_square(img: np.ndarray) -> np.ndarray:
    """crop vuông ở giữa ảnh"""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0+side, x0:x0+side]

def load_face_grayscale(path: str, size=(256, 256)) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot load image from {path}")
    
    img = center_crop_square(img)  
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def load_face_database(folder: str, size=(256, 256)) -> List[Tuple[str, np.ndarray]]:
    """
    load tất cả ảnh trong folder làm host face candidate.
    trả về list (filename, img_gray)
    """
    db = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = center_crop_square(img)  
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        db.append((fname, img))
    if not db:
        raise ValueError(f"no face images found in {folder}")
    return db


def simple_face_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    placeholder cho cost AAM (ross dùng AAM; ở đây mình dùng MSE để demo).
    """
    if a.shape != b.shape:
        raise ValueError("images must have same shape")
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff ** 2))


def select_n_hosts_by_similarity(
    private_face: np.ndarray,
    db: List[Tuple[str, np.ndarray]],
    n: int,
) -> List[Tuple[str, np.ndarray]]:
    """
    chọn n host face trong db có distance nhỏ nhất so với private_face.
    trả về list [(fname, img), ...]
    """
    scores = []
    for fname, img in db:
        d = simple_face_distance(private_face, img)
        scores.append((d, fname, img))
    scores.sort(key=lambda x: x[0])
    if len(scores) < n:
        raise ValueError(f"need at least {n} host faces in db")
    result = []
    for i in range(n):
        _, fname, img = scores[i]
        result.append((fname, img))
    return result

def normalize_faces_for_gevcs(
    host1: np.ndarray,
    host2: np.ndarray,
    private_face: np.ndarray,
    size=(256, 256)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    chuẩn hóa trước khi đưa vào GEVCS:
    - resize
    - đảm bảo grayscale
    (ross thật còn làm alignment theo landmark / AAM, ở đây mình simplify)
    """
    def _prep(img):
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        return cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)

    h1 = _prep(host1)
    h2 = _prep(host2)
    priv = _prep(private_face)
    return h1, h2, priv


def enrollment_generate_shares(
    private_face_path: str,
    host_db_folder: str,
    out_share1_path: str,
    out_share2_path: str,
    size=(256, 256),
):
    """
    phase đăng ký (enrollment) kiểu Ross:
    - load private face của user
    - load host face database
    - chọn 2 host gần private face nhất
    - normalize 3 ảnh
    - halftone
    - chạy GEVCS → share1, share2
    - lưu share ra disk (giả lập 2 server)
    """
    # 1) load private face
    private_face = load_face_grayscale(private_face_path, size=size)

    # 2) load db
    db = load_face_database(host_db_folder, size=size)

    # 3) chọn 2 host
    (name1, host1), (name2, host2) = select_two_hosts_by_similarity(private_face, db)
    print(f"selected hosts: {name1}, {name2}")

    # 4) normalize (ở đây chủ yếu resize)
    host1_n, host2_n, priv_n = normalize_faces_for_gevcs(host1, host2, private_face, size=size)

    # 5) optional: halftoning
    host1_h = floyd_steinberg_halftone(host1_n)
    host2_h = floyd_steinberg_halftone(host2_n)
    priv_h = floyd_steinberg_halftone(priv_n)

    # 6) GEVCS sinh share
    share1, share2 = gevcs_generate_shares(host1_h, host2_h, priv_h, 9)

    # 7) lưu share
    cv2.imwrite(out_share1_path, share1)
    cv2.imwrite(out_share2_path, share2)


def reconstruct_from_shares(share1_path: str, share2_path: str, out_path: str = None):
    """
    chồng 2 share để khôi phục face (dùng cho verify / debug).
    """
    s1 = cv2.imread(share1_path, cv2.IMREAD_GRAYSCALE)
    s2 = cv2.imread(share2_path, cv2.IMREAD_GRAYSCALE)
    if s1 is None or s2 is None:
        raise ValueError("cannot load one of the shares")
    rec = overlay_shares(s1, s2, op="and")
    if out_path is not None:
        cv2.imwrite(out_path, rec)
    return rec


def verify_user(
    share1_path: str,
    share2_path: str,
    claimed_face_path: str,
    size=(256, 256),
    threshold: float = 800.0,
) -> float:
    """
    verify rất đơn giản (demo):
    - chồng share1 + share2 → reconstructed face
    - so với claimed_face bằng distance
    - distance < threshold → ACCEPT, ngược lại REJECT

    trả về: distance score (càng nhỏ càng giống)
    """
    rec = reconstruct_from_shares(share1_path, share2_path)
    rec_resized = cv2.resize(rec, size, interpolation=cv2.INTER_AREA)
    claimed = load_face_grayscale(claimed_face_path, size=size)
    d = simple_face_distance(rec_resized, claimed)
    print(f"distance = {d:.2f}")
    if d < threshold:
        print("ACCEPT")
    else:
        print("REJECT")
    return d

def run_demo(
    scheme: str,
    private_face_path: str,
    host_db_folder: str,
    out_folder: str,
    size=(256, 256),
    m: int = 9,
    threshold: float = 800.0,
):
    """
    scheme: "2of2" hoặc "2of3"
    - đọc private_face + host_db
    - sinh shares
    - lưu ảnh share / reconstructed để xem
    """
    os.makedirs(out_folder, exist_ok=True)

    # 1) load private face + db
    private_face = load_face_grayscale(private_face_path, size=size)
    db = load_face_database(host_db_folder, size=size)

    # 2) halftone trước cho private
    priv_h = floyd_steinberg_halftone(private_face)
    # 2.5) lưu secret đã halftone để tiện so sánh
    cv2.imwrite(os.path.join(out_folder, "private_halftone.png"), priv_h)

    if scheme == "2of2":
        # chọn 2 host
        selected = select_n_hosts_by_similarity(private_face, db, 2)
        (name1, host1), (name2, host2) = selected
        print(f"[2of2] selected hosts: {name1}, {name2}")

        host1_h = floyd_steinberg_halftone(host1)
        host2_h = floyd_steinberg_halftone(host2)

        # sinh 2 share
        share1, share2 = gevcs_generate_shares(host1_h, host2_h, priv_h, m=m)
        
        # resize về cùng size để so MSE cho công bằng
        host1_h_resized = cv2.resize(host1_h, size, interpolation=cv2.INTER_AREA)
        host2_h_resized = cv2.resize(host2_h, size, interpolation=cv2.INTER_AREA)
        share1_resized  = cv2.resize(share1,  size, interpolation=cv2.INTER_AREA)
        share2_resized  = cv2.resize(share2,  size, interpolation=cv2.INTER_AREA)

        # 1) host so với secret (đã halftone)
        d_h1_priv = simple_face_distance(host1_h_resized, priv_h)
        d_h2_priv = simple_face_distance(host2_h_resized, priv_h)
        print(f"[2of2] d(host1_h, priv_h) = {d_h1_priv:.2f}")
        print(f"[2of2] d(host2_h, priv_h) = {d_h2_priv:.2f}")

        # 2) share so với host vs secret
        d_s1_h1   = simple_face_distance(share1_resized, host1_h_resized)
        d_s1_priv = simple_face_distance(share1_resized, priv_h)
        d_s2_h2   = simple_face_distance(share2_resized, host2_h_resized)
        d_s2_priv = simple_face_distance(share2_resized, priv_h)
        print(f"[2of2] share1 vs host1 = {d_s1_h1:.2f}, vs priv_h = {d_s1_priv:.2f}")
        print(f"[2of2] share2 vs host2 = {d_s2_h2:.2f}, vs priv_h = {d_s2_priv:.2f}")

        path_s1 = os.path.join(out_folder, "share1.png")
        path_s2 = os.path.join(out_folder, "share2.png")
        cv2.imwrite(path_s1, share1)
        cv2.imwrite(path_s2, share2)

        # overlay 1+2
        rec_12 = overlay_shares(share1, share2, op="and")
        path_rec = os.path.join(out_folder, "reconstructed_12.png")
        cv2.imwrite(path_rec, rec_12)

        # tính distance với secret đã halftone (priv_h) cho công bằng
        rec_resized = cv2.resize(rec_12, size, interpolation=cv2.INTER_AREA)
        # so sánh với cả host để tham khảo
        d_rec_priv  = simple_face_distance(rec_resized, priv_h)
        d_rec_host1 = simple_face_distance(rec_resized, host1_h_resized)
        d_rec_host2 = simple_face_distance(rec_resized, host2_h_resized)

        print(f"[2of2] d(rec_12, priv_h)  = {d_rec_priv:.2f}")
        print(f"[2of2] d(rec_12, host1_h) = {d_rec_host1:.2f}")
        print(f"[2of2] d(rec_12, host2_h) = {d_rec_host2:.2f}")
        # quyết định ACCEPT / REJECT
        d = simple_face_distance(rec_resized, priv_h)
        print(f"[2of2] distance(rec_12, priv_h) = {d:.2f}")
        if d < threshold:
            print("[2of2] ACCEPT")
        else:
            print("[2of2] REJECT")

    elif scheme == "2of3":
        # cần ít nhất 3 host trong db
        selected = select_n_hosts_by_similarity(private_face, db, 3)
        (name1, host1), (name2, host2), (name3, host3) = selected
        print(f"[2of3] selected hosts: {name1}, {name2}, {name3}")

        host1_h = floyd_steinberg_halftone(host1)
        host2_h = floyd_steinberg_halftone(host2)
        host3_h = floyd_steinberg_halftone(host3)

        # sinh 3 share
        share1, share2, share3 = gevcs_generate_shares_2of3(
            host1_h, host2_h, host3_h, priv_h, m=m
        )

        host1_h_resized = cv2.resize(host1_h, size, interpolation=cv2.INTER_AREA)
        host2_h_resized = cv2.resize(host2_h, size, interpolation=cv2.INTER_AREA)
        host3_h_resized = cv2.resize(host3_h, size, interpolation=cv2.INTER_AREA)

        share1_resized = cv2.resize(share1, size, interpolation=cv2.INTER_AREA)
        share2_resized = cv2.resize(share2, size, interpolation=cv2.INTER_AREA)
        share3_resized = cv2.resize(share3, size, interpolation=cv2.INTER_AREA)

        # 1) host so với secret (đã halftone)
        d_h1_priv = simple_face_distance(host1_h_resized, priv_h)
        d_h2_priv = simple_face_distance(host2_h_resized, priv_h)
        d_h3_priv = simple_face_distance(host3_h_resized, priv_h)
        print(f"[2of3] d(host1_h, priv_h) = {d_h1_priv:.2f}")
        print(f"[2of3] d(host2_h, priv_h) = {d_h2_priv:.2f}")
        print(f"[2of3] d(host3_h, priv_h) = {d_h3_priv:.2f}")

        # 2) share so với host vs secret (chỉ để tham khảo, không bắt buộc)
        d_s1_h1   = simple_face_distance(share1_resized, host1_h_resized)
        d_s1_priv = simple_face_distance(share1_resized, priv_h)
        d_s2_h2   = simple_face_distance(share2_resized, host2_h_resized)
        d_s2_priv = simple_face_distance(share2_resized, priv_h)
        d_s3_h3   = simple_face_distance(share3_resized, host3_h_resized)
        d_s3_priv = simple_face_distance(share3_resized, priv_h)
        print(f"[2of3] share1 vs host1 = {d_s1_h1:.2f}, vs priv_h = {d_s1_priv:.2f}")
        print(f"[2of3] share2 vs host2 = {d_s2_h2:.2f}, vs priv_h = {d_s2_priv:.2f}")
        print(f"[2of3] share3 vs host3 = {d_s3_h3:.2f}, vs priv_h = {d_s3_priv:.2f}")

        path_s1 = os.path.join(out_folder, "share1.png")
        path_s2 = os.path.join(out_folder, "share2.png")
        path_s3 = os.path.join(out_folder, "share3.png")
        cv2.imwrite(path_s1, share1)
        cv2.imwrite(path_s2, share2)
        cv2.imwrite(path_s3, share3)

        # overlay từng cặp
        rec_12 = overlay_shares(share1, share2, op="and")
        rec_13 = overlay_shares(share1, share3, op="and")
        rec_23 = overlay_shares(share2, share3, op="and")

        cv2.imwrite(os.path.join(out_folder, "reconstructed_12.png"), rec_12)
        cv2.imwrite(os.path.join(out_folder, "reconstructed_13.png"), rec_13)
        cv2.imwrite(os.path.join(out_folder, "reconstructed_23.png"), rec_23)

        # tính distance với secret đã halftone (priv_h) cho từng cặp
        for name, rec in [("12", rec_12), ("13", rec_13), ("23", rec_23)]:
            rec_resized = cv2.resize(rec, size, interpolation=cv2.INTER_AREA)
            d = simple_face_distance(rec_resized, priv_h)
            print(f"[2of3] distance(rec_{name}, priv_h) = {d:.2f}")
    else:
        raise ValueError("scheme must be '2of2' or '2of3'")

if __name__ == "__main__":
    private_face_path = "data/private_face.png"
    host_db_folder = "data/host_db"

    # demo (2,2): 2 share, phải chồng cả 2
    run_demo(
        scheme="2of2",
        private_face_path=private_face_path,
        host_db_folder=host_db_folder,
        out_folder="out_2of2",
        size=(256, 256),
        m=9,
        threshold= 23000.0,
    )

    # demo (2,3): 3 share, bất kỳ 2 share đều khôi phục được
    run_demo(
        scheme="2of3",
        private_face_path=private_face_path,
        host_db_folder=host_db_folder,
        out_folder="out_2of3",
        size=(256, 256),
        m=9,
        threshold=20000.0,
    )