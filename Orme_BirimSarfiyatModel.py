import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os
from datetime import datetime
from streamlit_gsheets import GSheetsConnection

# -----------------------------------------------------------------------------
# 1. AYARLAR VE DOSYA YÖNETİMİ
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Örme Sarfiyat Tahmini", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
EXCEL_NAME = "Orme_BirimSarfiyat_Yuklenecek.xlsx"
MODEL_NAME = "Orme_BirimSarfiyatModel.cbm"

excel_path = os.path.join(current_dir, EXCEL_NAME)
model_path = os.path.join(current_dir, MODEL_NAME)

@st.cache_data
def load_data():
    if not os.path.exists(excel_path):
        st.error(f"❌ Excel dosyası bulunamadı!")
        return None
    try:
        return pd.read_excel(excel_path)
    except Exception as e:
        st.error(f"Excel okuma hatası: {e}")
        return None

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası bulunamadı!")
        return None
    try:
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None

# Veri, Model ve Google Sheets Bağlantısı
df = load_data()
model = load_model()
conn = st.connection("gsheets", type=GSheetsConnection)

if df is None or model is None:
    st.stop()

# -----------------------------------------------------------------------------
# 2. ARAYÜZ VE FİLTRELEME
# -----------------------------------------------------------------------------
st.title("🧶 Örme Birim Sarfiyat Tahmini")
st.success("✅ Sistem Hazır. Değerleri girip hesapla butonuna basınız.")

inputs = {}
st.markdown("---")
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Model Seçimi")
    secilen_dept = st.selectbox("Departman", sorted(df['Departman'].astype(str).unique()))
    inputs['Departman'] = secilen_dept
    
    df_step1 = df[df['Departman'] == secilen_dept]
    secilen_tur = st.selectbox("Model_Turu", sorted(df_step1['Model_Turu'].astype(str).unique()))
    inputs['Model_Turu'] = secilen_tur
    
    df_step2 = df_step1[df_step1['Model_Turu'] == secilen_tur]
    secilen_detay = st.selectbox("Model_Detayi", sorted(df_step2['Model_Detayi'].astype(str).unique()))
    inputs['Model_Detayi'] = secilen_detay
    
    df_step3 = df_step2[df_step2['Model_Detayi'] == secilen_detay]
    secilen_fit = st.selectbox("Fit", sorted(df_step3['Fit'].astype(str).unique()))
    inputs['Fit'] = secilen_fit
    df_step4 = df_step3[df_step3['Fit'] == secilen_fit]

with col_right:
    st.subheader("⚙️ Teknik Detaylar")
    asorti_list = sorted(df_step4['Asorti'].astype(str).unique())
    if not asorti_list: asorti_list = sorted(df['Asorti'].astype(str).unique())
    inputs['Asorti'] = st.selectbox("Asorti", asorti_list)
    inputs['Pastal_Turu'] = st.selectbox("Pastal_Turu", sorted(df['Pastal_Turu'].astype(str).unique()))

    c1, c2 = st.columns(2)
    inputs['Kumas_Eni'] = c1.number_input("Kumas_Eni", 110.0, 200.0, 180.0)
    inputs['Kumas_Gramaji'] = c2.number_input("Kumas_Gramaji", 110.0, 420.0, 150.0)
    
    c3, c4 = st.columns(2)
    inputs['Toplam_Asorti'] = c3.number_input("Toplam_Asorti", 6.0, 14.0, 10.0)
    inputs['Parca_Sayisi'] = c4.number_input("Parca_Sayisi", 1.0, 13.0, 4.0)

# -----------------------------------------------------------------------------
# 3. HESAPLAMA VE KAYIT
# -----------------------------------------------------------------------------
st.divider()

if st.button("HESAPLA", type="primary", use_container_width=True):
    try:
        # Tahmin İşlemi
        X_new = pd.DataFrame([inputs])
        X_new = X_new[model.feature_names_]
        cat_features = ['Departman', 'Model_Turu', 'Model_Detayi', 'Fit', 'Pastal_Turu', 'Asorti']
        X_new_pool = Pool(X_new, cat_features=cat_features)
        prediction = model.predict(X_new_pool)[0]
        
        st.success(f"🧶 Tahmini Birim Sarfiyat: **{prediction:.3f} kg**")

        # --- GOOGLE SHEETS KAYIT BÖLÜMÜ ---
        # 1. Mevcut veriyi oku
        existing_data = conn.read(worksheet="Sheet1")
        
        # 2. Yeni satırı hazırla (Tarih ve Sonuç ekleyerek)
        new_row_data = inputs.copy()
        new_row_data['Tarih'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        new_row_data['Tahmin_Sonucu'] = round(prediction, 4)
        new_row_df = pd.DataFrame([new_row_data])
        
        # 3. Eski veriyle birleştir ve güncelle
        updated_df = pd.concat([existing_data, new_row_df], ignore_index=True)
        conn.update(worksheet="Sheet1", data=updated_df)
        
        st.info("📊 Tahmin verileri ve girişler Google Sheets'e kaydedildi.")

    except Exception as e:
        st.error(f"Hata oluştu: {e}")
