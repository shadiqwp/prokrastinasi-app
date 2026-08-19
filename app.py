import streamlit as st
import pandas as pd
import joblib
import gspread
import os
import json
from datetime import datetime
from google.oauth2.service_account import Credentials

# =========================
# UI CONFIG (Must be first)
# =========================
st.set_page_config(
    page_title="Klasifikasi Prokrastinasi Akademik",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI
st.markdown("""
<style>
    .glass-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
    }
    .glass-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .glass-header p {
        margin-top: 0.5rem;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    .info-box {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# CONFIG & LOAD MODEL
# =========================
SHEET_NAME = "hasil_responden_pass"
WORKSHEET_NAME = "Sheet1"
LOCAL_JSON = "service_account.json"

@st.cache_resource
def load_models():
    return (
        joblib.load("model_decision_tree_id3_final.pkl"),
        joblib.load("fitur_model_a.pkl"),
        joblib.load("area_mapping.pkl"),
        joblib.load("reverse_columns.pkl")
    )

model, fitur_model, area_mapping, reverse_columns = load_models()

# =========================
# GOOGLE SHEETS & SECRETS
# =========================
def connect_to_gsheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    if os.path.exists(LOCAL_JSON):
        creds = Credentials.from_service_account_file(LOCAL_JSON, scopes=scopes)
    elif "GCP_CREDENTIALS" in os.environ:
        try:
            creds_dict = json.loads(os.environ["GCP_CREDENTIALS"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        except json.JSONDecodeError:
            st.error("🚨 Format GCP_CREDENTIALS di environment variable tidak valid (harus JSON).")
            st.stop()
    elif "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    else:
        st.error("🚨 Kredensial GCP tidak ditemukan! Pastikan file `service_account.json` tersedia.")
        st.stop()

    client = gspread.authorize(creds)
    return client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)


def simpan_ke_google_sheets(data):
    sheet = connect_to_gsheet()
    existing = sheet.get_all_values()
    if len(existing) == 0:
        sheet.append_row(list(data.keys()))
    sheet.append_row(list(data.values()))

# =========================
# LOGIC FUNCTIONS
# =========================
def get_faktor_personal(data_processed, top_n=3):
    data_series = pd.Series(data_processed)
    top_faktor = data_series.sort_values(ascending=False).head(top_n)
    
    return [
        {"Kode": kode, "Keterangan": area_mapping.get(kode, kode), "Skor": int(skor)}
        for kode, skor in top_faktor.items()
    ]

def generate_kesimpulan(tingkat, faktor_personal):
    faktor_list = [item["Keterangan"] for item in faktor_personal]
    if len(faktor_list) == 1:
        faktor_text = faktor_list[0]
    elif len(faktor_list) == 2:
        faktor_text = f"{faktor_list[0]} dan {faktor_list[1]}"
    else:
        faktor_text = ", ".join(faktor_list[:-1]) + f", dan {faktor_list[-1]}"

    if tingkat == "Rendah":
        return f"Tingkat prokrastinasi akademik **Rendah**. Area yang masih perlu diperhatikan adalah **{faktor_text}**."
    elif tingkat == "Sedang":
        return f"Tingkat prokrastinasi akademik **Sedang**. Area dominan penundaan adalah **{faktor_text}**."
    else:
        return f"Tingkat prokrastinasi akademik **Tinggi**. Area dominan adalah **{faktor_text}**. Perlu perhatian ekstra terhadap manajemen waktu."

# =========================
# LAYOUT & UI
# =========================
st.markdown("""
<div class="glass-header">
    <h1>🎓 Klasifikasi Prokrastinasi Akademik</h1>
    <p>Sistem Deteksi Dini Berbasis Model Decision Tree ID3</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3076/3076404.png", width=80)
    st.header("ℹ️ Petunjuk Pengisian")
    st.markdown("""
    Pilih skala yang paling sesuai dengan kondisi Anda:
    - **1** = Sangat Rendah / Tidak Pernah
    - **2** = Rendah / Jarang
    - **3** = Sedang / Kadang-kadang
    - **4** = Tinggi / Sering
    - **5** = Sangat Tinggi / Selalu
    """)
    st.info("💡 Prokrastinasi akademik adalah perilaku menunda-nunda tugas akademik yang dapat mengakibatkan penurunan prestasi belajar dan kesejahteraan mental mahasiswa.")

st.markdown("""
<div class="info-box">
    <strong>Selamat Datang!</strong> Silakan isi identitas dan kuesioner PASS (Procrastination Assessment Scale-Students) di bawah ini untuk mengetahui tingkat penundaan akademik Anda.
</div>
""", unsafe_allow_html=True)

# FORM
with st.form("form_pass"):
    st.subheader("👤 Identitas Responden")
    
    # Baris 1: Nama, Email, NIM
    col_id1, col_id2, col_id3 = st.columns(3)
    with col_id1: nama = st.text_input("Nama Lengkap")
    with col_id2: email = st.text_input("Email")
    with col_id3: nim = st.text_input("NIM")

    # Baris 2: Jurusan, Semester, Jenis Kelamin
    col_id4, col_id5, col_id6 = st.columns(3)
    with col_id4: 
        jurusan = st.selectbox("Jurusan", [
            "Teknik Sipil", 
            "Teknik Kimia", 
            "Teknik Elektro", 
            "Teknik Mesin", 
            "Arsitektur"
        ])
    with col_id5: 
        semester = st.selectbox("Semester", ["1", "2", "3", "4", "5", "6", "7", "8", ">8"])
    with col_id6: 
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

    st.divider()
    st.subheader("📝 Kuesioner (A1 - A18)")
    
    jawaban_asli = {}
    
    col_q1, col_q2 = st.columns(2)
    for i, fitur in enumerate(fitur_model):
        pertanyaan = area_mapping.get(fitur, fitur)
        col_target = col_q1 if i % 2 == 0 else col_q2
        with col_target:
            st.markdown(f"**{fitur}. {pertanyaan.capitalize()}**")
            jawaban_asli[fitur] = st.radio(
                label=f"Skala {fitur}",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True,
                label_visibility="collapsed",
                key=fitur
            )
            st.write("") 

    st.divider()
    
    st.subheader("🌟 Evaluasi Sistem")
    feedback_berguna = st.radio(
        "Apakah menurut Anda website klasifikasi ini berguna?", 
        ["Ya", "Tidak"], 
        horizontal=True
    )

    st.write("")
    submitted = st.form_submit_button("🔍 Proses & Analisis Data", use_container_width=True)

# =========================
# SUBMIT LOGIC
# =========================
if submitted:
    # 1. Validasi Input Kosong
    if not nama.strip() or not email.strip() or not nim.strip():
        st.error("⚠️ Nama, Email, dan NIM wajib diisi!")
        st.stop()
        
    # 2. Validasi Format Email
    if "@" not in email:
        st.error("⚠️ Format Email tidak valid! Pastikan email mengandung karakter '@'.")
        st.stop()
        
    # 3. Validasi NIM harus angka
    if not nim.isdigit():
        st.error("⚠️ NIM hanya boleh berisi angka (tanpa spasi atau huruf)!")
        st.stop()

    with st.spinner("Menganalisis data responden..."):
        data_processed = jawaban_asli.copy()
        
        # Reverse logic
        for col in reverse_columns:
            if col in data_processed:
                data_processed[col] = 6 - data_processed[col]

        # Prediction
        input_df = pd.DataFrame([data_processed])[fitur_model]
        prediksi = model.predict(input_df)[0]
        total_skor = sum(data_processed.values())
        faktor_personal = get_faktor_personal(data_processed, top_n=3)
        kesimpulan = generate_kesimpulan(prediksi, faktor_personal)

        # Save to GS (Ubah "prodi" menjadi "jurusan")
        data_simpan = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nama_kode": nama,
            "email": email,
            "nim": nim,
            "jurusan": jurusan,
            "semester": semester,
            "jenis_kelamin": jenis_kelamin,
        }
        for f in fitur_model: data_simpan[f] = jawaban_asli[f]
        
        data_simpan.update({
            "total_skor_processed": total_skor,
            "tingkat_prokrastinasi": prediksi,
            "faktor_dominan_1": f"{faktor_personal[0]['Kode']} - {faktor_personal[0]['Keterangan']}",
            "skor_faktor_1": faktor_personal[0]["Skor"],
            "faktor_dominan_2": f"{faktor_personal[1]['Kode']} - {faktor_personal[1]['Keterangan']}",
            "skor_faktor_2": faktor_personal[1]["Skor"],
            "faktor_dominan_3": f"{faktor_personal[2]['Kode']} - {faktor_personal[2]['Keterangan']}",
            "skor_faktor_3": faktor_personal[2]["Skor"],
            "website_berguna": feedback_berguna
        })

        try:
            simpan_ke_google_sheets(data_simpan)
        except Exception as e:
            st.error("Gagal menyimpan ke Database (Google Sheets).")
            st.code(str(e))
            st.stop()

    # Show Results
    st.success("✅ Data berhasil diproses dan disimpan!")
    st.divider()
    
    st.markdown("### 📊 Laporan Hasil Klasifikasi")
    
    r_col1, r_col2 = st.columns([1, 1.5])
    
    with r_col1:
        # Mengganti tag <div> HTML menjadi st.container untuk menghindari error layout tabel kosong
        with st.container(border=True):
            st.metric(label="Total Skor PASS", value=total_skor)
            
            if prediksi == "Rendah":
                st.success("Tingkat: **RENDAH** 🟢")
            elif prediksi == "Sedang":
                st.warning("Tingkat: **SEDANG** 🟡")
            else:
                st.error("Tingkat: **TINGGI** 🔴")

    with r_col2:
        st.markdown("#### 🎯 Faktor Dominan (3 Tertinggi)")
        for idx, item in enumerate(faktor_personal):
            st.write(f"{idx+1}. **{item['Kode']}** — {item['Keterangan']} (Skor: {item['Skor']})")
            st.progress(item['Skor'] / 5.0)

    st.markdown("#### 📝 Rekomendasi / Kesimpulan")
    st.info(kesimpulan)