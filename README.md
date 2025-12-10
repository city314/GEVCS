# GEVCS for Face Biometrics (Ross-style)

Repo này demo **Gray-level Extended Visual Cryptography Scheme (GEVCS)** cho ảnh mặt, dựa trên ý tưởng của Ross (biometric + visual crypto).

Có 2 chế độ:

- `(2,2)` – 2 share, chồng cả 2 mới khôi phục secret  

---

## 1. Cấu trúc repo

```text
.
├── biometric_ross.py    # script chính: load data, chạy demo, log kết quả
├── gevcs_core.py        # core GEVCS: halftone + generate share + overlay
├── data/
│   ├── private_face.png # ảnh chủ (secret)
│   ├── host1.png        # host 1
│   ├── host2.png        # host 2
│   ├── host3.png        # host 3
│   └── host_db/         # folder chứa host (nếu dùng dạng DB)
└── out_2of2/ # thư mục kết quả demo
```

---

## 2. Môi trường & thư viện

Tối thiểu:

- Python 3.8+
- `numpy`
- `opencv-python` (`cv2`)
- `Pillow` (nếu cần đọc/ghi ảnh kiểu khác)

Cài nhanh:

```bash
pip install numpy opencv-python pillow
```

---

## 3. Chuẩn bị dữ liệu

Thư mục `data/`:

- `private_face.png` : ảnh mặt chủ cần bảo vệ (secret)
- `host1.png`, `host2.png`, `host3.png` : ảnh host dùng để “giấu” secret
- các ảnh nên:
  - khuôn mặt nhìn thẳng (frontal)
  - nền tương đối đơn giản
  - crop gần vuông, độ phân giải đủ lớn (nhưng không quá nhoè)

Code sẽ tự resize về đúng size khi chạy (mặc định `256×256`).

---

## 4. Tiền xử lý & metric

### 4.1. Tiền xử lý ảnh

Trong `biometric_ross.py`:

1. **Load & resize**

   - Tất cả ảnh (private + host) được load dạng grayscale và resize về kích thước cố định, ví dụ:

     ```python
     size = (256, 256)
     ```

2. **Halftone (Floyd–Steinberg)**

   - Ảnh xám được chuyển sang dạng **halftone đen–trắng** bằng thuật toán Floyd–Steinberg:

     ```python
     priv_h   # secret sau halftone
     host1_h  # host1 sau halftone
     host2_h  # ...
     host3_h  # ...
     ```

   - Đây mới là ảnh thật sự tham gia vào GEVCS.  
     `private_face.png` chỉ để nhìn cho dễ, còn “secret” trong scheme là `priv_h`.

### 4.2. Metric: simple_face_distance

- Hàm `simple_face_distance(a, b)` (trong `biometric_ross.py`) dùng **MSE trên ảnh halftone/grayscale**:

  > d(a, b) = MSE(a, b)

- Tất cả các log kiểu:

  - `d(host1_h, priv_h)`
  - `distance(rec_12, priv_h)`
  - `share1 vs host1 = ..., vs priv_h = ...`

  đều dùng metric này.

---

## 5. Baseline của dữ liệu

Với bộ 4 ảnh trong `data/` hiện tại (private + 3 host), ta đo:

- `d(host1_h, priv_h) ≈ 32584`
- `d(host2_h, priv_h) ≈ 33103`
- `d(host3_h, priv_h) ≈ 32681`

Có thể xem đây là:

> **baseline impostor distance** của bộ dữ liệu:  
> khoảng cách giữa **secret** và **một người khác bất kỳ** ≈ **32k–33k** (sau halftone, MSE).

Baseline này dùng để:

- so xem **reconstructed** có *gần secret hơn host* hay không
- chọn **ngưỡng (threshold)** cho bước xác thực (ACCEPT / REJECT)

---

## 6. Cách chạy demo

### 6.1. Demo (2,2)

Trong `biometric_ross.py`, phần `if __name__ == "__main__":` đã có sẵn lời gọi:

```python
run_demo(
    scheme="2of2",
    private_face_path="data/private_face.png",
    host_db_folder="data/host_db",
    out_folder="out_2of2",
    size=(256, 256),
    m=9,               # số subpixel (kích thước block)
    threshold=23000.0, # ngưỡng verify đề xuất cho (2,2)
)
```

Chạy:

```bash
python biometric_ross.py
```

Kết quả:

- Thư mục `out_2of2/` sẽ có:
  - `private_halftone.png`  – secret sau halftone
  - `host1_halftone.png`, `host2_halftone.png`, ...
  - `share1.png`, `share2.png`
  - `reconstructed_12.png` – overlay share1 + share2

- Log trên terminal sẽ in các distance:

  ```text
  [2of2] d(host1_h, priv_h) = ...
  [2of2] d(host2_h, priv_h) = ...
  [2of2] share1 vs host1 = ..., vs priv_h = ...
  [2of2] share2 vs host2 = ..., vs priv_h = ...
  [2of2] d(rec_12, priv_h)  = ...
  [2of2] d(rec_12, host1_h) = ...
  [2of2] d(rec_12, host2_h) = ...
  [2of2] distance(rec_12, priv_h) = ...
  [2of2] ACCEPT / REJECT
  ```

**Ý nghĩa (baseline):**

- host vs secret: ~32k (impostor)  
- reconstructed vs secret: ~17k–21k  
- threshold 2of2 đề xuất: **~23k**

→  

- `distance(rec_12, priv_h) < 23k` → ACCEPT  
- `d(host*_h, priv_h) > 23k` → REJECT



## 7. Cách đọc log để biết scheme có “đúng” không

### 7.1. Share có “leak secret” nhiều không?

Nhìn vào:

```text
share1 vs host1 = A, vs priv_h = B
```

- nếu `A << B` → share1 giống host hơn → **riêng tư hơn**
- nếu `B << A` → share1 giống secret hơn → **leak nhiều hơn**

Trong config hiện tại:

- 2of2: 1 share nghiêng về host, 1 share hơi nghiêng về secret  
- 2of3: cả 3 share đều gần secret hơn host (demo rõ mặt, privacy 1-share chưa mạnh, phù hợp demo hơn là production).

### 7.2. Reconstructed có “giống secret hơn host” không?

Check:

```text
d(rec_ij, priv_h)  ?
d(rec_ij, host*_h) ?
```

- nếu `d(rec_ij, priv_h) < d(rec_ij, host*_h)` cho mọi host  
  → overlay share đã **kéo về secret**, scheme ok.

Đồng thời phải:

- `d(rec_ij, priv_h)` **nhỏ hơn nhiều** baseline 32k  
- `d(host*_h, priv_h)` ≈ 32k

---

## 8. Chỉnh tham số

Một vài tham số quan trọng trong `gevcs_core.py`:

- `m` – số subpixel trong mỗi block (kích thước tile)
  - m lớn → ảnh share/reconstructed mịn hơn nhưng file nặng hơn.
- `nb_secret` – số subpixel dành cho secret (tỷ lệ với `d_secret**0.8 * m`)
  - tăng hệ số (0.5 → 0.7 → 1.0):
    - reconstructed rõ secret hơn
    - share giống secret hơn (privacy giảm)
  - giảm hệ số:
    - share gần host hơn (privacy tăng)
    - reconstructed mờ hơn, distance với secret tăng

Khi đổi tham số / đổi dữ liệu, nên:

1. Đo lại `d(host*_h, priv_h)` để lấy **baseline mới**.  
2. Đo `d(rec_ij, priv_h)` với config mới.  
3. Chọn lại threshold sao cho:

   ```text
   max( d(rec_ij, priv_h) ) < threshold < min( d(host*_h, priv_h) )
   ```

---