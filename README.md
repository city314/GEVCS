# GEVCS for Face Biometrics (Ross-style) + Deep Host Selection

repo này demo **Gray-level Extended Visual Cryptography Scheme (GEVCS)** cho ảnh mặt, dựa trên ý tưởng của Ross (biometric + visual crypto), và thêm bước **chọn host bằng deep learning (ArcFace / DeepFace)**.

có 2 chế độ:

- `(2,2)` – 2 share, chồng cả 2 mới khôi phục secret  
- `(2,3)` – 3 share, **bất kỳ 2 share** chồng lên đều khôi phục được secret  

deep learning chỉ dùng ở **bước chọn host**; còn phần GEVCS (halftone + generate share + overlay) vẫn giống bản gốc.

---

## 1. cấu trúc repo

```text
.
├── biometric_ross.py    # script chính: load data, chọn host (deep), chạy demo, log kết quả
├── gevcs_core.py        # core GEVCS: halftone + generate share + overlay
├── data/
│   ├── private_face.png # ảnh chủ (secret)
│   ├── host1.png        # host 1
│   ├── host2.png        # host 2
│   ├── host3.png        # host 3
│   └── host_db/         # folder chứa host (nếu muốn mở rộng DB)
└── out_2of2/, out_2of3/ # thư mục kết quả demo
```

---

## 2. môi trường & thư viện

tối thiểu:

- Python 3.8+
- `numpy`
- `opencv-python` (`cv2`)
- `Pillow`
- **`deepface`** (gồm ArcFace model + TensorFlow ở backend)

cài nhanh:

```bash
pip install numpy opencv-python pillow deepface
```

lần chạy đầu `deepface` sẽ tự tải trọng số ArcFace về `~/.deepface/weights/`, nên hơi lâu 1 chút.

---

## 3. chuẩn bị dữ liệu

thư mục `data/`:

- `private_face.png` : ảnh mặt chủ cần bảo vệ (secret)
- `host1.png`, `host2.png`, `host3.png` : các ảnh host
- yêu cầu nhẹ:
  - khuôn mặt nhìn thẳng (frontal)
  - nền tương đối đơn giản
  - crop gần vuông, đủ rõ (không quá mờ)

code sẽ tự resize về kích thước cố định (mặc định `256×256`).

---

## 4. tiền xử lý, deep host selection & metric

### 4.1. tiền xử lý ảnh

trong `biometric_ross.py`:

1. **load & resize**

   - tất cả ảnh (private + host) load dạng grayscale, resize về `size = (256, 256)`.

2. **halftone (Floyd–Steinberg)**

   - ảnh xám chuyển sang **halftone đen–trắng**:

     ```python
     priv_h   # secret sau halftone
     host1_h  # host1 sau halftone
     host2_h  # ...
     host3_h  # ...
     ```

   - secret thật sự trong scheme chính là `priv_h` chứ không phải ảnh gốc.

### 4.2. deep host selection (ArcFace / DeepFace)

bước mới so với bản cũ:

- dùng `DeepFace` với model **`ArcFace`** để trích xuất **embedding khuôn mặt**.

trong code:

```python
from deepface import DeepFace

def face_embedding_from_gray(img_gray):
    # img_gray: ảnh mặt grayscale 256×256
    # convert sang RGB rồi cho vào DeepFace.represent(...)
    reps = DeepFace.represent(
        img_rgb,
        model_name="ArcFace",
        enforce_detection=False,  # ảnh đã crop sẵn
    )
    emb = reps[0]["embedding"]  # vector float32 ~512 chiều
```

hàm chọn host:

```python
def select_n_hosts_by_deep(private_face, db, n):
    priv_emb = face_embedding_from_gray(private_face)
    # với mỗi host trong db:
    #   - tính embedding
    #   - tính L2 distance giữa priv_emb và host_emb
    # chọn ra n host gần nhất
```

- với `(2,2)`: chọn 2 host gần nhất  
- với `(2,3)`: chọn 3 host gần nhất  

**lưu ý**: deep learning chỉ dùng để chọn host. đánh giá scheme (ACCEPT / REJECT) vẫn dùng metric MSE trên ảnh halftone (mục 4.3).

### 4.3. metric: simple_face_distance (MSE halftone)

hàm `simple_face_distance(a, b)` dùng **MSE trên ảnh halftone / grayscale**:

> `d(a, b) = MSE(a, b)`

tất cả các log:

- `d(host*_h, priv_h)`
- `share1 vs host1 = ..., vs priv_h = ...`
- `distance(rec_12, priv_h) = ...`

đều dùng metric này.

---

## 5. baseline của dữ liệu (MSE)

với bộ 4 ảnh trong `data/` (private + 3 host), sau halftone:

- `d(host1_h, priv_h) ≈ 33k`
- `d(host2_h, priv_h) ≈ 33k`
- `d(host3_h, priv_h) ≈ 33k` (dao động 32k–33k)

coi như:

> **baseline impostor distance** (sau halftone, MSE)  
> khoảng cách giữa **secret** và **một người khác** ≈ **32k–33k**.

baseline này dùng để:

- so xem ảnh **reconstructed** có gần secret hơn host không  
- chọn **ngưỡng (threshold)** cho bước xác thực (ACCEPT / REJECT)

deep host selection không thay baseline MSE này; nó chỉ đảm bảo host được chọn **giống mặt** secret hơn (theo embedding), nên share nhìn trực quan hợp lý hơn.

---

## 6. cách chạy demo

### 6.1. demo (2,2)

trong `biometric_ross.py`:

```python
run_demo(
    scheme="2of2",
    private_face_path="data/private_face.png",
    host_db_folder="data/host_db",
    out_folder="out_2of2",
    size=(256, 256),
    m=9,               # số subpixel mỗi block
    threshold=23000.0, # ngưỡng verify đề xuất cho (2,2)
)
```

chạy:

```bash
python biometric_ross.py
```

kết quả:

- `out_2of2/` chứa:
  - `private_halftone.png`
  - `host*_halftone.png`
  - `share1.png`, `share2.png`
  - `reconstructed_12.png`

log trên terminal (ví dụ):

```text
[2of2] selected hosts: host1.png, host2.png   # chọn bằng deep (ArcFace)
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

**diễn giải nhanh:**

- host vs secret ≈ 32k–33k (impostor)  
- reconstructed vs secret ≈ 15k–21k  
- threshold gợi ý: **~23k**

→  

- `distance(rec_12, priv_h) < 23k` → ACCEPT  
- `d(host*_h, priv_h) > 23k` → REJECT

### 6.2. demo (2,3)

tương tự:

```python
run_demo(
    scheme="2of3",
    private_face_path="data/private_face.png",
    host_db_folder="data/host_db",
    out_folder="out_2of3",
    size=(256, 256),
    m=9,
    threshold=20000.0,  # ngưỡng đề xuất cho (2,3)
)
```

chạy:

```bash
python biometric_ross.py
```

kết quả:

- `out_2of3/` có:
  - `private_halftone.png`
  - `share1.png`, `share2.png`, `share3.png`
  - `reconstructed_12.png`, `reconstructed_13.png`, `reconstructed_23.png`

log:

```text
[2of3] selected hosts: host1.png, host2.png, host3.png   # chọn bằng deep

[2of3] d(host1_h, priv_h) = ...
[2of3] d(host2_h, priv_h) = ...
[2of3] d(host3_h, priv_h) = ...

[2of3] share1 vs host1 = ..., vs priv_h = ...
[2of3] share2 vs host2 = ..., vs priv_h = ...
[2of3] share3 vs host3 = ..., vs priv_h = ...

[2of3] distance(rec_12, priv_h) = ...
[2of3] distance(rec_13, priv_h) = ...
[2of3] distance(rec_23, priv_h) = ...
```

**baseline 2of3:**

- host vs secret: ~32k–33k  
- rec_ij vs secret: ~8k–10k  
- threshold gợi ý: **15k–20k** (code đang dùng ~20k)

---

## 7. cách đọc log (kèm deep host selection)

### 7.1. host được deep chọn có hợp lý không?

- dòng:

  ```text
  selected hosts: host1.png, host2.png, ...
  ```

  cho biết những host có embedding ArcFace gần secret nhất.

- nếu nhìn ảnh halftone mà thấy:
  - pose, góc mặt, nền khá giống secret → deep selection đang hoạt động tốt.

### 7.2. share có “leak secret” nhiều không?

xem:

```text
share1 vs host1 = A, vs priv_h = B
```

- `A << B` → share1 giống host hơn, riêng tư hơn  
- `B << A` → share1 giống secret hơn, leak nhiều hơn

tuỳ mục tiêu demo / privacy mà chỉnh tham số trong `gevcs_core.py` cho phù hợp.

### 7.3. reconstructed có “giống secret hơn host” không?

check:

```text
d(rec_ij, priv_h)  ?
d(rec_ij, host*_h) ?
```

- yêu cầu tối thiểu:

  ```text
  d(rec_ij, priv_h) < d(rec_ij, host*_h)     # reconstruct kéo về secret
  d(rec_ij, priv_h) << 32k                   # nhỏ hơn nhiều baseline impostor
  d(host*_h, priv_h) ≈ 32k                   # host khác người
  ```

---

## 8. chỉnh tham số & hướng mở rộng

- trong `gevcs_core.py`:
  - `m`: số subpixel/block. tăng m → ảnh mịn hơn nhưng file nặng hơn.
  - hệ số trong tính `nb_secret` điều khiển mức “trộn” secret vào host:
    - tăng hệ số → reconstructed rõ hơn, share giống secret hơn  
    - giảm hệ số → share an toàn hơn, reconstructed mờ hơn

- có thể mở rộng:

  - dùng model khác của deepface (`Facenet512`, `VGG-Face`, …) cho bước chọn host  
  - dùng trực tiếp **khoảng cách embedding** làm metric verify (thay vì MSE) nếu muốn gần với hệ face recognition hơn.
