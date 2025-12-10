# GEVCS biometric (2-of-2)

this repo contains a small demo for a **gray-level extended visual cryptography scheme (gevcs)** applied to **biometric privacy** for faces (2-of-2 scheme).

a private face image is encoded into **2 shares (share1, share2)** that look like normal host images / noise. only when you stack the 2 shares together you can reconstruct the private face. each single share alone should not reveal the identity.

the code follows the idea of ross & othman (2011) and nakajima & yamaguchi (extended vcs for natural images), but is implemented in a simple python pipeline.

---

## 1. project layout

```text
GEVCS/
├── data/
│   ├── host_db/          # database of host faces
│   ├── private_face.png  # private face for demo
│   └── out_2of2/         # outputs from cli demo
│
├── app.py                # streamlit web ui demo
├── biometric_ross_v2.py  # main gevcs 2-of-2 pipeline (cli)
├── deep_face_utils.py    # helper for deep face embeddings
├── gevcs_core_v2.py      # core gevcs operations (halftone, shares, reconstruct)
├── perceptual_features.py# vgg16 perceptual features for host selection
├── simple_evaluator.py   # simple metrics (mse / variance, etc.)
├── vcs_evaluator.py      # evaluator using insightface embeddings
├── share1.png            # example share1
├── share2.png            # example share2
├── temp_private.png      # aligned private face used by app
├── requirements.txt      # python dependencies
└── README.md             # this file
```

---

## 2. installation

### 2.1 create environment

recommend using python 3.9+ and a virtual environment.

```bash
python -m venv .venv
# linux / macos
source .venv/bin/activate
# windows
# .venv\Scripts\activate
```

### 2.2 install dependencies

all external libraries are listed in `requirements.txt`. this includes things like torch, torchvision, insightface, onnxruntime, streamlit, etc.

just run:

```bash
pip install -r requirements.txt
```

this will install every third‑party package used by the **local modules** in this repo (`gevcs_core_v2.py`, `vcs_evaluator.py`, `perceptual_features.py`, ...).  
the local files themselves are part of the repo, so you do **not** pip‑install them separately.

if something is still missing, you can also install basic packages manually:

```bash
pip install opencv-python numpy pillow
```

---

## 3. data preparation

inside the `data/` folder:

- `private_face.png`  
  - face image used as the secret. can be gray or color; the code will convert to gray, align, and resize to 256x256.

- `host_db/`  
  - folder of host face images. each file is one host. the pipeline will:
    - center‑crop / resize to 256x256
    - align faces using insightface
    - compute perceptual features to select the best hosts

- `out_2of2/`  
  - output folder for the cli demo. after running the demo you will see:
    - `share1.png`
    - `share2.png`
    - `reconstructed.png`
    - optional intermediate/eval images

you can replace `private_face.png` with your own image and add more images into `host_db/` to try different examples.

---

## 4. running the cli demo (biometric_ross_v2.py)

`biometric_ross_v2.py` implements the 2-of-2 gevcs pipeline.

default entry:

```python
if __name__ == "__main__":
    run_demo(
        private_face_path="data/private_face.png",
        host_db_folder="data/host_db",
        out_folder="data/out_2of2",
        size=(256, 256),
        m=16,
    )
```

to run:

```bash
python biometric_ross_v2.py
```

what it does:

1. **preprocess & align**
   - load `private_face.png` and all hosts from `host_db/`
   - convert to grayscale, center‑crop, resize to 256x256
   - align faces using insightface (buffalo_l model)

2. **select 2 hosts**
   - extract vgg16 perceptual features for each host and the private face
   - compute l2 distances and pick the 2 closest hosts

3. **generate shares (2-of-2 gevcs)**
   - perform gray‑level preprocessing and halftoning
   - encode the private face into 2 shares based on host images (parameter `m` is the number of subpixels per pixel, default 16)

4. **reconstruction**
   - overlay share1 and share2 to reconstruct the face
   - apply enhancement (normalization, contrast, sharpening) to improve visual quality

5. **evaluation**
   - use `vcs_evaluator.py` with insightface embeddings
   - compute:
     - distance between private face and each host (host similarity)
     - distance between private face and reconstructed image (reconstruction quality)
   - (leakage for each share is currently set to 0 in this version, i.e., we assume no identity from a single share)

all resulting images are saved into `data/out_2of2/`.

note: first run may be slow while models (vgg16 and buffalo_l) are downloaded.

---

## 5. running the streamlit app (app.py)

`app.py` provides a simple web ui for the same pipeline.

run:

```bash
streamlit run app.py
```

main steps inside the ui:

1. **upload private face**
   - upload an image file
   - the app shows:
     - original image
     - aligned + resized grayscale face
     - halftone / binary version used by gevcs

2. **load host_db and generate shares**
   - specify folder path for `host_db` (default `data/host_db`)
   - app:
     - previews some host images
     - automatically selects 2 hosts based on vgg16 perceptual distance
     - generates `share1` and `share2` using the gevcs core

3. **view shares**
   - displays `share1` and `share2` side by side
   - each share should look like a normal host / noisy image and not reveal the secret alone

4. **reconstruction**
   - overlays share1 and share2 to reconstruct the face
   - shows the enhanced reconstruction (256x256)

5. **evaluation**
   - computes the same distances as in the cli demo
   - prints:
     - host1 distance
     - host2 distance
     - share1 leakage
     - share2 leakage
     - reconstruction distance

---

## 6. notes

- default resolution is 256x256; you can change `size` and `m` in `biometric_ross_v2.py` and `app.py` if you want to experiment.
- gpu is optional but recommended if you run many images; otherwise cpu is still ok but slower.
- this code is for study / research demo only, not production security.

