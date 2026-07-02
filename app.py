import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from parser import (
    extract_text_from_pdf,
    extract_text_from_docx,
    parse_cv_with_gemini,
    get_embedding,
    calculate_cosine_similarity,
)
from db import init_db, save_job_posting, save_candidate

init_db()

st.set_page_config(page_title="TalentAI", page_icon="🎯", layout="wide")
st.title("TalentAI — CV Analiz & Eşleştirme")

col1, col2 = st.columns(2)

with col1:
    st.subheader("İş İlanı")
    position = st.text_input("Pozisyon Adı", placeholder="örn. Senior Python Developer")
    job_description = st.text_area("İş Tanımı", height=150, placeholder="Görev tanımı ve sorumluluklar...")
    requirements = st.text_area("Aranan Nitelikler", height=150, placeholder="Teknik beceriler, deneyim, sertifikalar...")

    if st.button("İlan Vektörünü Hesapla", use_container_width=True):
        if not job_description.strip() and not requirements.strip():
            st.warning("Lütfen en az iş tanımı veya aranan nitelikler girin.")
        else:
            with st.spinner("Vektör hesaplanıyor..."):
                job_text = "\n\n".join(filter(None, [position, job_description, requirements]))
                job_vector = get_embedding(job_text)
                save_job_posting(position or "İsimsiz Pozisyon", job_description, requirements, job_vector)
                st.session_state["job_vector"] = job_vector
            st.success("İlan vektörü hesaplandı ve veritabanına kaydedildi.")

with col2:
    st.subheader("CV Yükleme & Analiz")
    uploaded_files = st.file_uploader(
        "CV Yükle (PDF veya DOCX)", type=["pdf", "docx"], accept_multiple_files=True
    )

    if st.button("CV'leri Analiz Et ve Eşleştir", use_container_width=True):
        if not uploaded_files:
            st.warning("Lütfen önce en az bir CV dosyası yükleyin.")
        elif "job_vector" not in st.session_state:
            st.warning("İlan vektörü bulunamadı. Sol sütunda iş ilanını girin ve 'İlan Vektörünü Hesapla' butonuna basın.")
        else:
            results = []
            for uploaded_file in uploaded_files:
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                with st.spinner(f"{uploaded_file.name}: metin çıkarılıyor..."):
                    if ext == ".pdf":
                        cv_text = extract_text_from_pdf(tmp_path)
                    else:
                        cv_text = extract_text_from_docx(tmp_path)

                os.unlink(tmp_path)

                if not cv_text.strip():
                    st.error(f"{uploaded_file.name}: CV'den metin çıkarılamadı.")
                    continue

                with st.spinner(f"{uploaded_file.name}: Gemini ile analiz ediliyor..."):
                    cv_data = parse_cv_with_gemini(cv_text)

                with st.spinner(f"{uploaded_file.name}: vektörleştiriliyor..."):
                    cv_vector = get_embedding(cv_text)
                    save_candidate(uploaded_file.name, cv_data, cv_text, cv_vector)

                score = calculate_cosine_similarity(st.session_state["job_vector"], cv_vector)
                results.append((uploaded_file.name, score, cv_data))

            if results:
                results.sort(key=lambda r: r[1], reverse=True)

                st.subheader("Eşleşme Sonuçları")
                result_cols = st.columns(len(results))
                for col, (name, score, cv_data) in zip(result_cols, results):
                    with col:
                        st.markdown(f"**{name}**")
                        if score >= 75:
                            st.success(f"%{score:.1f} — Yüksek Uyum")
                        elif score >= 50:
                            st.info(f"%{score:.1f} — Orta Uyum")
                        else:
                            st.warning(f"%{score:.1f} — Düşük Uyum")
                        with st.expander("Ayrıştırılmış CV Verisi"):
                            st.json(cv_data)

                if results[0][1] >= 75:
                    st.balloons()
