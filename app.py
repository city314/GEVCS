import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

from biometric_ross_v2 import (
    load_face_database,
    align_face,
    select_n_hosts_by_similarity,
    enrollment_generate_shares,
    reconstruct_from_shares
)
from gevcs_core_v2 import floyd_steinberg_halftone
from vcs_evaluator import VCSEvaluator


# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="GEVCS Biometric Demo — Minimal", layout="wide")
st.title("🔐 GEVCS Biometric Demo — Minimal UI")


# -------------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------------
def init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

init("private_aligned", None)
init("db", [])
init("selected_hosts", None)
init("share1_path", "share1.png")
init("share2_path", "share2.png")
init("reconstructed", None)


# -------------------------------------------------------
# 1️⃣ Upload Private Face
# -------------------------------------------------------
st.header("1️⃣ Upload Private Face")

uploaded = st.file_uploader("Chọn ảnh khuôn mặt (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded)
    st.image(pil_img, caption="Ảnh Private Gốc (Màu)", width=250)

    img = pil_img.convert("L")
    private = np.array(img)


    # Align face
    aligned = align_face(private)
    st.session_state.private_aligned = aligned

    st.image(aligned, caption="Ảnh Private đã Align", width=250)

    # Halftone (demo)
    private_halftone = floyd_steinberg_halftone(aligned)
    st.image(private_halftone, caption="Ảnh Halftone của Private", width=250)

elif st.session_state.private_aligned is not None:
    st.subheader("Ảnh Private đã lưu")
    st.image(st.session_state.private_aligned, width=250)


# -------------------------------------------------------
# 2️⃣ Load Host Database → Auto generate shares
# -------------------------------------------------------
st.header("2️⃣ Load Host Database & Generate Shares")

host_folder = st.text_input("Nhập đường dẫn host_db:", value="data/host_db")

if st.button("📂 Load & Generate Shares"):
    st.info("📁 Đang load host database...")

    # Prevent loading before uploading private
    if st.session_state.private_aligned is None:
        st.error("❌ Bạn phải upload ảnh PRIVATE trước!")
    elif not os.path.isdir(host_folder):
        st.error(f"❌ Folder '{host_folder}' không tồn tại!")
    else:
        st.session_state.db = load_face_database(host_folder)
        # Hiển thị 4 host đầu tiên để demo
        st.subheader("Một số ảnh Host trong Database:")
        cols = st.columns(4)
        for i, (name, host_img) in enumerate(st.session_state.db[:4]):
            cols[i].image(host_img, caption=name, width=150)

        st.info("🔎 Đang chọn host theo deep similarity...")
        # Auto select host (KHÔNG hiển thị ảnh host nữa)
        st.session_state.selected_hosts = select_n_hosts_by_similarity(
            st.session_state.private_aligned,
            st.session_state.db,
            n=2
        )

        # Auto generate shares
        cv2.imwrite("temp_private.png", st.session_state.private_aligned)
        st.info("✨ Đang sinh shares từ 2 host được chọn...")

        enrollment_generate_shares(
            private_face_path="temp_private.png",
            host_db_folder=host_folder,
            out1=st.session_state.share1_path,
            out2=st.session_state.share2_path,
            size=(256,256),
            m=16
        )

        st.success("🎉 Shares đã được tạo!")


# -------------------------------------------------------
# Display Shares ONLY
# -------------------------------------------------------
st.header("📌 Shares Output")

if st.session_state.selected_hosts is not None:

    if os.path.exists(st.session_state.share1_path):
        s1 = cv2.imread(st.session_state.share1_path, cv2.IMREAD_GRAYSCALE)
        st.image(s1, caption="Share 1", width=250)

    if os.path.exists(st.session_state.share2_path):
        s2 = cv2.imread(st.session_state.share2_path, cv2.IMREAD_GRAYSCALE)
        st.image(s2, caption="Share 2", width=250)


# -------------------------------------------------------
# 3️⃣ Reconstruction
# -------------------------------------------------------
st.header("3️⃣ Reconstruction")

if st.button("🧩 Tái tạo ảnh từ shares"):
    rec = reconstruct_from_shares(
        st.session_state.share1_path,
        st.session_state.share2_path
    )
    st.session_state.reconstructed = rec
    st.success("Ảnh tái tạo thành công!")

if st.session_state.reconstructed is not None:
    st.image(st.session_state.reconstructed, caption="Ảnh tái tạo", width=300)


# -------------------------------------------------------
# 4️⃣ Evaluation
# -------------------------------------------------------
st.header("4️⃣ Evaluation")

if st.button("📊 Evaluate"):
    if st.session_state.reconstructed is None:
        st.error("❌ Chưa có ảnh tái tạo để đánh giá!")
    else:
        evaluator = VCSEvaluator()

        (h1_name, h1_img), (h2_name, h2_img) = st.session_state.selected_hosts

        result = evaluator.evaluate_all(
            st.session_state.private_aligned,
            h1_img,
            h2_img,
            st.session_state.reconstructed
        )

        st.subheader("🔎 Evaluation Result")
        st.write(f"Host1 distance: {result['host1_distance']:.3f}")
        st.write(f"Host2 distance: {result['host2_distance']:.3f}")
        st.write(f"Share1 leakage: {result['share1_leakage']}")
        st.write(f"Share2 leakage: {result['share2_leakage']}")
        st.write(f"Reconstruction distance: {result['reconstruction_distance']:.3f}")

        st.image(result["reconstructed_img"], caption="Ảnh tái tạo sau enhance", width=300)
