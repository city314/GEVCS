import numpy as np
import cv2
from insightface.app import FaceAnalysis


class DeepFaceEmbedder:
    def __init__(self, det_size=(128, 128), ctx_id=0):
        """
        ctx_id=0: dùng GPU nếu có
        ctx_id=-1: dùng CPU
        """
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def get_embedding(self, img: np.ndarray):
        """
        Lấy embedding (512-d vector) từ ảnh.
        Nếu không detect được mặt → trả về None.
        """
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img

        faces = self.app.get(img_bgr)
        if len(faces) == 0:
            return None

        # chọn mặt lớn nhất (phòng ảnh có nhiều mặt)
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        emb = faces[0].embedding.astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-8)
        return emb


def deep_face_distance(emb1, emb2):
    """
    Khoảng cách L2 giữa 2 embedding.
    Embedding càng gần → mặt càng giống.
    """
    if emb1 is None or emb2 is None:
        return 1e9
    return float(np.linalg.norm(emb1 - emb2))
