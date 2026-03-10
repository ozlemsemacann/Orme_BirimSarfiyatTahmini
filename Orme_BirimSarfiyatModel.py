import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os

# 1. AYARLAR VE DOSYA YÖNETİMİ
st.set_page_config(page_title="Örme Sarfiyat Tahmini", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
EXCEL_NAME = "Orme_BirimSarfiyat_Yuklenecek.xlsx"
MODEL_NAME = "Orme_BirimSarfiyatModel.cbm"

excel_path = os.path.join(current_dir, EXCEL_NAME)
model_path = os.path.join(current_dir, MODEL_NAME)

@st.cache_data
def load_data():
    if not os.path.exists(excel_path):
        st.error(f"❌ Excel dosyası bulunamadı! Lütfen '{EXCEL_NAME}' adında bir dosyayı proje klasörüne yükle.")
        return None
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        st.error(f"Excel okuma hatası: {e}")
        return None

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası bulunamadı! ({MODEL_NAME})")
        return None
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

df = load_data()
model = load_model()

if df is None:
    st.stop()

# 2. FİLTRELEME VE OTOMATİK HESAPLAMA
st.title("🧶 Örme Birim Sarfiyat Tahmini (Otomatik Parametreli)")
st.markdown("---")

inputs = {}
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Model Seçimi")
    
    # Kademeli Filtreleme
    dept_list = sorted(df['Departman'].unique())
    secilen_dept = st.selectbox("Departman", dept_list)
    df_step1 = df[df['Departman'] == secilen_dept]

    tur_list = sorted(df_step1['Model_Turu'].unique())
    secilen_tur = st.selectbox("Model Türü", tur_list)
    df_step2 = df_step1[df_step1['Model_Turu'] == secilen_tur]

    detay_list = sorted(df_step1['Model_Detayi'].unique())
    secilen_tur = st.selectbox("Model Fetayi", tur_list)
    df_step2 = df_step2[df_step2['Model_Detayi'] == secilen_tur]

    fit_list = sorted(df_step2['Fit'].unique())
    secilen_fit = st.selectbox("Fit", fit_list)
    df_step4 = df_step3[df_step3['Fit'] == secilen_fit]

    # --- OTOMATİK PARÇA SAYISI HESAPLAMA ---
    # Seçilen kriterlere uyan verilerin ortalama parça sayısını al
    if not df_step3.empty:
        otomatik_parca = round(df_step3['Parca_Sayisi'].mean())
    else:
        otomatik_parca = round(df['Parca_Sayisi'].mean())
    
    st.info(f"💡 Seçilen modele göre önerilen Parça Sayısı: **{otomatik_parca}**")
    inputs['Parca_Sayisi'] = otomatik_parca

with col_right:
    st.subheader("⚙️ Teknik Detaylar")

    # ASORTİ SEÇİMİ
    asorti_list = sorted(df_step3['Asorti'].unique())
    if not asorti_list:
        asorti_list = sorted(df['Asorti'].unique())
    
    secilen_asorti_adi = st.selectbox("Asorti Tipi", asorti_list)
    inputs['Asorti'] = secilen_asorti_adi

    # --- OTOMATİK TOPLAM ASORTİ HESAPLAMA ---
    # Seçilen asorti adına karşılık gelen sayısal Toplam_Asorti değerini bul
    toplam_asorti_degeri = df[df['Asorti'] == secilen_asorti_adi]['Toplam_Asorti'].iloc[0]
    st.info(f"🔢 Seçilen asortinin toplam içeriği: **{toplam_asorti_degeri}**")
    inputs['Toplam_Asorti'] = toplam_asorti_degeri

    inputs['Pastal_Turu'] = st.selectbox("Pastal Türü", sorted(df['Pastal_Turu'].unique()))

    c1, c2 = st.columns(2)
    inputs['Kumas_Eni'] = c1.number_input("Kumaş Eni", 110.0, 200.0, 180.0)
    inputs['Kumas_Gramaji'] = c2.number_input("Kumaş Gramajı", 110.0, 420.0, 150.0)

# 3. HESAPLAMA
st.divider()

# Modelin beklediği diğer sütunları (Departman vb.) inputs sözlüğüne ekle
inputs['Departman'] = secilen_dept
inputs['Model_Turu'] = secilen_tur
inputs['Fit'] = secilen_fit

if st.button("HESAPLA", type="primary", use_container_width=True):
    if model:
        try:
            # DataFrame oluştur
            X_new = pd.DataFrame([inputs])
            
            # Modelin eğitimdeki sütun sırasını zorla (Hata almamak için) 
            beklenen_siralama = model.feature_names_
            X_new = X_new[beklenen_siralama]

            # Kategorik değişkenleri modelin tanıması için listele
            cat_features = [col for col in X_new.columns if X_new[col].dtype == 'object']
            
            X_new_pool = Pool(X_new, cat_features=cat_features)
            prediction = model.predict(X_new_pool)[0]
            
            st.success(f"🧶 Tahmini Birim Sarfiyat: **{prediction:.3f} kg**")
            
            # Seçilen otomatik değerleri özetle
            st.write(f"*(Hesaplamada kullanılan otomatik değerler: Parça Sayısı: {inputs['Parca_Sayisi']}, Toplam Asorti: {inputs['Toplam_Asorti']})*")
            
        except Exception as e:
            st.error(f"Hata: {e}")
    else:
        st.error("Model dosyası yüklenemediği için hesaplama yapılamıyor.")

