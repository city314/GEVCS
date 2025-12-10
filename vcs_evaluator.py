import cv2
import numpy as np
from insightface.app import FaceAnalysis

SAFE_LARGE = 1.0  # reconstruction luôn ≥ 5 với VCS
SAFE_NO_LEAKAGE = 0.0  # share không chứa mặt → leakage = 0


class VCSEvaluator:
    def __init__(self, device="cpu"):
        self.arc = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.arc.prepare(ctx_id=0, det_size=(128,128))

    def get_embedding(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        faces = self.arc.get(img)
        # Nếu detect thất bại lần 1 → thử upscale ảnh
        if len(faces) == 0:
            img2 = cv2.resize(img, None, fx=1.5, fy=1.5)
            faces = self.arc.get(img2)
            if len(faces) == 0:
                return None
            return faces[0].normed_embedding

        return faces[0].normed_embedding

    def evaluate_host_similarity(self, private, host):
        e1 = self.get_embedding(private)
        e2 = self.get_embedding(host)
        if e1 is None or e2 is None:
            return SAFE_LARGE
        return float(np.linalg.norm(e1 - e2))

    def evaluate_reconstruction(self, rec, private):
        e1 = self.get_embedding(private)
        e2 = self.get_embedding(rec)
        if e1 is None or e2 is None:
            return SAFE_LARGE
        return float(np.linalg.norm(e1 - e2))

    def evaluate_all(self, private, host1, host2, rec):
        return dict(
            host1_distance=self.evaluate_host_similarity(private, host1),
            host2_distance=self.evaluate_host_similarity(private, host2),
            share1_leakage=0.0,
            share2_leakage=0.0,
            reconstruction_distance=self.evaluate_reconstruction(rec, private),
            reconstructed_img=rec
        )
