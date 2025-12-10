import numpy as np
import cv2

def mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def contains_face_like_structure(img, threshold=50.0):
    """
    Bản GE-VCS cơ bản không có share leakage mạnh,
    nhưng ta có thể ước lượng "có hình hay không"
    bằng cách đo độ biến thiên (variance) trong ảnh.

    - variance thấp → toàn trắng / toàn đen → KHÔNG có mặt
    - variance cao  → có cấu trúc → CÓ thể rò rỉ mặt
    """
    v = np.var(img.astype(np.float32))  # measure structure

    return v > threshold, v


class SimpleVCSEvaluator:
    """
    Đánh giá GE-VCS cơ bản:
    - Host similarity (MSE với private)
    - Share leakage (share có cấu trúc giống mặt?)
    - Reconstruction quality (MSE với private)
    """

    def evaluate_host_quality(self, private, host):
        return mse(private, host)

    def evaluate_share_leakage(self, share):
        has_face_like, variance = contains_face_like_structure(share)
        return has_face_like, variance

    def evaluate_reconstruction_quality(self, private, reconstructed):
        return mse(private, reconstructed)

    def evaluate_all(self, private, host1, host2, share1, share2, reconstructed):
        results = {}

        # Host quality
        results["host1_mse"] = self.evaluate_host_quality(private, host1)
        results["host2_mse"] = self.evaluate_host_quality(private, host2)

        # Share leakage
        leak1, var1 = self.evaluate_share_leakage(share1)
        leak2, var2 = self.evaluate_share_leakage(share2)

        results["share1_leakage"] = leak1
        results["share1_variance"] = var1
        results["share2_leakage"] = leak2
        results["share2_variance"] = var2

        # Reconstruction
        results["reconstruction_mse"] = self.evaluate_reconstruction_quality(private, reconstructed)

        return results
