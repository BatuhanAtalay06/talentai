import os
import tempfile
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors as genai_errors

load_dotenv()

from parser import (
    extract_text_from_pdf,
    extract_text_from_docx,
    parse_cv_with_gemini,
    get_embedding,
    calculate_cosine_similarity,
)
from db import (
    init_db,
    save_job_posting,
    save_candidate,
    list_candidates,
    list_job_postings,
    get_candidate,
    update_candidate,
    delete_candidate,
    get_job_posting,
    update_job_posting,
    delete_job_posting,
    save_match,
    list_matches,
    delete_match,
)


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429:
            return "Gemini API kullanım limitine ulaşıldı. Lütfen birkaç dakika bekleyip tekrar deneyin."
        if exc.code == 404:
            return "Gemini modeli bulunamadı. Model adı değişmiş veya kaldırılmış olabilir."
        if exc.code and exc.code >= 500:
            return "Gemini servisinde geçici bir sorun oluştu. Lütfen birkaç saniye sonra tekrar deneyin."
        return f"Gemini isteği başarısız oldu: {exc.message or exc}"
    if isinstance(exc, psycopg2.OperationalError):
        return "Veritabanına bağlanılamadı. Postgres'in çalıştığından emin olun (örn. 'docker compose ps')."
    if isinstance(exc, psycopg2.Error):
        return "Veritabanı işlemi sırasında bir hata oluştu."
    return f"Beklenmeyen bir hata oluştu: {exc}"


st.set_page_config(page_title="TalentAI", page_icon="🎯", layout="wide")
st.title("TalentAI — CV Analiz & Eşleştirme")

try:
    init_db()
except Exception as e:
    st.error(f"Uygulama başlatılamadı — {friendly_error(e)}")
    st.stop()

tab_eslestirme, tab_kisiler, tab_ilanlar, tab_gecmis = st.tabs(
    ["Eşleştirme", "Kayıtlı Kişiler", "İş İlanları", "Eşleşme Geçmişi"]
)

with tab_eslestirme:
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
                try:
                    with st.spinner("Vektör hesaplanıyor..."):
                        job_text = "\n\n".join(filter(None, [position, job_description, requirements]))
                        job_vector = get_embedding(job_text)
                        job_posting_id = save_job_posting(position or "İsimsiz Pozisyon", job_description, requirements, job_vector)
                        st.session_state["job_vector"] = job_vector
                        st.session_state["job_posting_id"] = job_posting_id
                        st.session_state["job_position_name"] = position or "İsimsiz Pozisyon"
                    st.success("İlan vektörü hesaplandı ve veritabanına kaydedildi.")
                except Exception as e:
                    st.error(friendly_error(e))

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

                    try:
                        with st.spinner(f"{uploaded_file.name}: metin çıkarılıyor..."):
                            if ext == ".pdf":
                                cv_text = extract_text_from_pdf(tmp_path)
                            else:
                                cv_text = extract_text_from_docx(tmp_path)
                    except Exception as e:
                        st.error(f"{uploaded_file.name}: metin çıkarılamadı — {friendly_error(e)}")
                        continue
                    finally:
                        os.unlink(tmp_path)

                    if not cv_text.strip():
                        st.error(f"{uploaded_file.name}: CV'den metin çıkarılamadı.")
                        continue

                    try:
                        with st.spinner(f"{uploaded_file.name}: Gemini ile analiz ediliyor..."):
                            cv_data = parse_cv_with_gemini(cv_text)

                        with st.spinner(f"{uploaded_file.name}: vektörleştiriliyor..."):
                            cv_vector = get_embedding(cv_text)
                            candidate_id = save_candidate(uploaded_file.name, cv_data, cv_text, cv_vector)
                    except Exception as e:
                        st.error(f"{uploaded_file.name}: {friendly_error(e)}")
                        continue

                    score = calculate_cosine_similarity(st.session_state["job_vector"], cv_vector)
                    try:
                        save_match(
                            st.session_state.get("job_posting_id"),
                            candidate_id,
                            st.session_state.get("job_position_name", "İsimsiz Pozisyon"),
                            cv_data.get("ad_soyad") or uploaded_file.name,
                            score,
                        )
                    except Exception as e:
                        st.warning(f"{uploaded_file.name}: eşleşme geçmişine kaydedilemedi — {friendly_error(e)}")

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

with tab_kisiler:
    st.subheader("Kayıtlı Kişiler")
    if "candidate_flash" in st.session_state:
        st.success(st.session_state.pop("candidate_flash"))
    try:
        candidates = list_candidates()
    except Exception as e:
        st.error(friendly_error(e))
        candidates = []

    if not candidates:
        st.info("Henüz kayıtlı kişi yok. 'Eşleştirme' sekmesinden CV yükleyip analiz ettiğinizde burada listelenecek.")
    else:
        st.caption(f"Toplam {len(candidates)} kişi")
        table_rows = [
            {
                "Ad Soyad": c["ad_soyad"] or "-",
                "E-posta": c["e_posta"] or "-",
                "Telefon": c["telefon"] or "-",
                "Deneyim (yıl)": c["deneyim_yili"],
                "Yetenekler": ", ".join(c["yetenekler"] or []),
                "Eğitim": ", ".join(c["egitim"] or []),
                "Dosya": c["file_name"],
                "Kayıt Tarihi": c["created_at"].strftime("%Y-%m-%d %H:%M"),
            }
            for c in candidates
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        with st.expander("Özetleri Gör"):
            for c in candidates:
                st.markdown(f"**{c['ad_soyad'] or c['file_name']}**")
                st.write(c["ozet"] or "_Özet yok._")
                st.divider()

        st.divider()
        st.subheader("Kişi Yönet")
        candidate_options = {f"{c['ad_soyad'] or c['file_name']} (#{c['id']})": c["id"] for c in candidates}
        selected_candidate_label = st.selectbox(
            "Düzenlenecek / silinecek kişi", list(candidate_options.keys()), key="candidate_select"
        )
        selected_candidate_id = candidate_options[selected_candidate_label]
        candidate = get_candidate(selected_candidate_id)

        with st.form("edit_candidate_form"):
            ad_soyad_edit = st.text_input("Ad Soyad", value=candidate["ad_soyad"] or "")
            e_posta_edit = st.text_input("E-posta", value=candidate["e_posta"] or "")
            telefon_edit = st.text_input("Telefon", value=candidate["telefon"] or "")
            deneyim_yili_edit = st.number_input(
                "Deneyim (yıl)", value=float(candidate["deneyim_yili"] or 0), min_value=0.0, step=0.5
            )
            yetenekler_edit = st.text_area("Yetenekler (virgülle ayırın)", value=", ".join(candidate["yetenekler"] or []))
            egitim_edit = st.text_area("Eğitim (virgülle ayırın)", value=", ".join(candidate["egitim"] or []))
            ozet_edit = st.text_area("Özet", value=candidate["ozet"] or "")

            if st.form_submit_button("Kaydet"):
                try:
                    update_candidate(
                        selected_candidate_id,
                        ad_soyad_edit,
                        e_posta_edit,
                        telefon_edit,
                        deneyim_yili_edit,
                        [s.strip() for s in yetenekler_edit.split(",") if s.strip()],
                        [s.strip() for s in egitim_edit.split(",") if s.strip()],
                        ozet_edit,
                    )
                    st.session_state["candidate_flash"] = "Kişi güncellendi."
                    st.rerun()
                except Exception as e:
                    st.error(friendly_error(e))

        confirm_delete_candidate = st.checkbox("Silmeyi onaylıyorum", key="confirm_delete_candidate")
        if st.button("Kişiyi Sil", disabled=not confirm_delete_candidate):
            try:
                delete_candidate(selected_candidate_id)
                st.session_state["candidate_flash"] = "Kişi silindi."
                st.rerun()
            except Exception as e:
                st.error(friendly_error(e))

with tab_ilanlar:
    st.subheader("İş İlanları")
    if "job_flash" in st.session_state:
        st.success(st.session_state.pop("job_flash"))
    try:
        job_postings = list_job_postings()
    except Exception as e:
        st.error(friendly_error(e))
        job_postings = []

    if not job_postings:
        st.info("Henüz kayıtlı iş ilanı yok. 'Eşleştirme' sekmesinden ilan girip 'İlan Vektörünü Hesapla' butonuna bastığınızda burada listelenecek.")
    else:
        st.caption(f"Toplam {len(job_postings)} ilan")
        table_rows = [
            {
                "Pozisyon": j["position"],
                "İş Tanımı": j["description"] or "-",
                "Aranan Nitelikler": j["requirements"] or "-",
                "Kayıt Tarihi": j["created_at"].strftime("%Y-%m-%d %H:%M"),
            }
            for j in job_postings
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("İlan Yönet")
        job_options = {f"{j['position']} (#{j['id']})": j["id"] for j in job_postings}
        selected_job_label = st.selectbox("Düzenlenecek / silinecek ilan", list(job_options.keys()), key="job_select")
        selected_job_id = job_options[selected_job_label]
        job = get_job_posting(selected_job_id)

        with st.form("edit_job_form"):
            position_edit = st.text_input("Pozisyon Adı", value=job["position"])
            description_edit = st.text_area("İş Tanımı", value=job["description"] or "", height=120)
            requirements_edit = st.text_area("Aranan Nitelikler", value=job["requirements"] or "", height=120)

            if st.form_submit_button("Kaydet (vektör yeniden hesaplanır)"):
                try:
                    with st.spinner("Vektör yeniden hesaplanıyor..."):
                        job_text_edit = "\n\n".join(filter(None, [position_edit, description_edit, requirements_edit]))
                        new_vector = get_embedding(job_text_edit)
                        update_job_posting(selected_job_id, position_edit, description_edit, requirements_edit, new_vector)
                    st.session_state["job_flash"] = "İlan güncellendi."
                    st.rerun()
                except Exception as e:
                    st.error(friendly_error(e))

        confirm_delete_job = st.checkbox("Silmeyi onaylıyorum", key="confirm_delete_job")
        if st.button("İlanı Sil", disabled=not confirm_delete_job):
            try:
                delete_job_posting(selected_job_id)
                st.session_state["job_flash"] = "İlan silindi."
                st.rerun()
            except Exception as e:
                st.error(friendly_error(e))

with tab_gecmis:
    st.subheader("Eşleşme Geçmişi")
    if "match_flash" in st.session_state:
        st.success(st.session_state.pop("match_flash"))
    try:
        matches = list_matches()
    except Exception as e:
        st.error(friendly_error(e))
        matches = []

    if not matches:
        st.info("Henüz eşleşme geçmişi yok. 'Eşleştirme' sekmesinde bir analiz çalıştırdığınızda burada listelenecek.")
    else:
        st.caption(f"Toplam {len(matches)} eşleşme")

        def uyum_label(score):
            if score >= 75:
                return "Yüksek"
            if score >= 50:
                return "Orta"
            return "Düşük"

        table_rows = [
            {
                "Kişi": m["candidate_name"],
                "İlan": m["job_position"],
                "Skor (%)": round(float(m["score"]), 1),
                "Uyum": uyum_label(float(m["score"])),
                "Tarih": m["created_at"].strftime("%Y-%m-%d %H:%M"),
            }
            for m in matches
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Geçmiş Yönet")
        match_options = {
            f"{m['candidate_name']} → {m['job_position']} (%{round(float(m['score']), 1)}, #{m['id']})": m["id"]
            for m in matches
        }
        selected_match_label = st.selectbox("Silinecek kayıt", list(match_options.keys()), key="match_select")
        selected_match_id = match_options[selected_match_label]

        confirm_delete_match = st.checkbox("Silmeyi onaylıyorum", key="confirm_delete_match")
        if st.button("Kaydı Sil", disabled=not confirm_delete_match):
            try:
                delete_match(selected_match_id)
                st.session_state["match_flash"] = "Kayıt silindi."
                st.rerun()
            except Exception as e:
                st.error(friendly_error(e))
