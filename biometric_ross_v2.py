import os
import cv2
import numpy as np
from typing import List, Tuple

from vcs_evaluator import VCSEvaluator
from perceptual_features import PerceptualExtractor, perceptual_distance

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

os.environ["INSIGHTFACE_DISABLE_LOG"] = "1"

from gevcs_core_v2 import (
    floyd_steinberg_halftone,
    gevcs_generate_shares_ross,
    overlay_shares,
    enhance_reconstruction,
)


# ============================
# 1. Utility
# ============================

def center_crop_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]


def load_face_grayscale(path, size=(256, 256)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = center_crop_square(img)
    return cv2.resize(img, size)


def load_face_database(folder, size=(256, 256)):
    db = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = center_crop_square(img)
        img = cv2.resize(img, size)
        db.append((fname, img))
    return db


# ============================
# 2. Align
# ============================

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(128,128))

def align_face(img, size=256):
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        rgb = img

    faces = face_app.get(rgb)
    if len(faces) == 0:
        print("[WARN] No face detected → fallback to resize")
        return cv2.resize(img, (size,size))

    face = faces[0]
    aligned = norm_crop(rgb, face.kps, image_size=size)
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)


# ============================
# 3. Host selection (VGG)
# ============================

def select_n_hosts_by_similarity(private_face, db, n=2):
    extractor = PerceptualExtractor(device="cpu")
    f_priv = extractor.get_feat(private_face)

    scores = []
    for fname, img in db:
        f_host = extractor.get_feat(img)
        d = perceptual_distance(f_priv, f_host)
        scores.append((d, fname, img))

    scores.sort(key=lambda x: x[0])
    return [(name, img) for (_, name, img) in scores[:n]]


# ============================
# 4. Enrollment
# ============================

def enrollment_generate_shares(private_face_path, host_db_folder,
                               out1, out2, size=(256,256), m=16):

    priv = load_face_grayscale(private_face_path, size)
    db = load_face_database(host_db_folder, size)

    priv_al = align_face(priv)
    db_al = [(n, align_face(im)) for (n, im) in db]

    (n1, h1), (n2, h2) = select_n_hosts_by_similarity(priv_al, db_al, n=2)
    print("Selected hosts:", n1, n2)

    h1_d = cv2.bilateralFilter(h1, 7, 50, 50)
    h2_d = cv2.bilateralFilter(h2, 7, 50, 50)
    priv_d = cv2.bilateralFilter(priv_al, 7, 50, 50)

    s1, s2 = gevcs_generate_shares_ross(
        host1_gray=h1_d,
        host2_gray=h2_d,
        secret_gray=priv_d,
        m=m,
        L=0.55,
    )

    cv2.imwrite(out1, s1)
    cv2.imwrite(out2, s2)


# ============================
# 5. Reconstruction
# ============================

def reconstruct_from_shares(p1, p2, out=None):
    s1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    s2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

    rec = overlay_shares(s1, s2)
    rec = enhance_reconstruction(rec)

    if out:
        cv2.imwrite(out, rec)
    return rec


# ============================
# 6. Demo
# ============================

def run_demo(private_face_path, host_db_folder, out_folder,
             size=(256, 256), m=16):

    os.makedirs(out_folder, exist_ok=True)

    priv = load_face_grayscale(private_face_path)
    db = load_face_database(host_db_folder)

    priv_al = align_face(priv)
    db_al = [(n, align_face(im)) for (n, im) in db]

    (n1, h1), (n2, h2) = select_n_hosts_by_similarity(priv_al, db_al)
    print("Selected hosts:", n1, n2)

    # Halftone debug
    priv_h = floyd_steinberg_halftone(priv_al)
    cv2.imwrite(out_folder + "/private_halftone_debug.png", priv_h)

    # Preprocessing
    h1_d = cv2.bilateralFilter(h1, 7, 50, 50)
    h2_d = cv2.bilateralFilter(h2, 7, 50, 50)
    p_d = cv2.bilateralFilter(priv_al, 7, 50, 50)

    # Generate shares
    s1, s2 = gevcs_generate_shares_ross(
        host1_gray=h1_d,
        host2_gray=h2_d,
        secret_gray=p_d,
        m=m,
        L=0.55)

    cv2.imwrite(out_folder + "/share1.png", s1)
    cv2.imwrite(out_folder + "/share2.png", s2)

    # Reconstruction
    rec = overlay_shares(s1, s2)
    rec = enhance_reconstruction(rec)
    cv2.imwrite(out_folder + "/reconstructed.png", rec)

    # Evaluation
    evaluator = VCSEvaluator(device="cpu")
    results = evaluator.evaluate_all(priv_al, h1, h2, rec)

    print("\n===== HOST QUALITY =====")
    print("Host1 distance:", results["host1_distance"])
    print("Host2 distance:", results["host2_distance"])

    print("\n===== SHARE LEAKAGE =====")
    print("Share1 leakage:", results["share1_leakage"])
    print("Share2 leakage:", results["share2_leakage"])

    print("\n===== RECONSTRUCTION QUALITY =====")
    print("Reconstruction:", results["reconstruction_distance"])

    cv2.imwrite(out_folder + "/reconstructed_eval.png", results["reconstructed_img"])


if __name__ == "__main__":
    run_demo(
        private_face_path="data/private_face.png",
        host_db_folder="data/host_db",
        out_folder="out_2of2",
        size=(256, 256),
        m=16,
    )
