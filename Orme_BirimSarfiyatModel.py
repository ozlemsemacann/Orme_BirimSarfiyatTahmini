import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os

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
    try:
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None

df = load_data()
model = load_model()

if df is None or model is None:
    st.stop()

# -----------------------------------------------------------------------------
# 2. BASAMAKLI FİLTRELEME (CASCADING FILTERS)
# -----------------------------------------------------------------------------
st.title("🧶 Örme Birim Sarfiyat Tahmini")
st.success(f"✅ Model ve Veri Hazır! Seçimleri yaparak tahmini alabilirsiniz.")

inputs = {}
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Model Seçimi")

    # 1. DEPARTMAN
    dept_list = sorted(df['Departman'].astype(str).unique())
    secilen_dept = st.selectbox("Departman", dept_list)
    inputs['Departman'] = secilen_dept
    df_step1 = df[df['Departman'] == secilen_dept]

    # 2. MODEL TÜRÜ
    tur_list = sorted(df_step1['Model_Turu'].astype(str).unique())
    secilen_tur = st.selectbox("Model_Turu", tur_list)
    inputs['Model_Turu'] = secilen_tur
    df_step2 = df_step1[df_step1['Model_Turu'] == secilen_tur]

    # 3. MODEL DETAYI
    detay_list = sorted(df_step2['Model_Detayi'].astype(str).unique())
    secilen_detay = st.selectbox("Model_Detayi", detay_list)
    inputs['Model_Detayi'] = secilen_detay
    df_step3 = df_step2[df_step2['Model_Detayi'] == secilen_detay]

    # 4. FIT
    fit_list = sorted(df_step3['Fit'].astype(str).unique())
    secilen_fit = st.selectbox("Fit", fit_list)
    inputs['Fit'] = secilen_fit
    df_step4 = df_step3[df_step3['Fit'] == secilen_fit]

with col_right:
    st.subheader("⚙️ Teknik Detaylar")

    # 5. ASORTI
    asorti_list = sorted(df_step4['Asorti'].astype(str).unique())
    if not asorti_list:
        asorti_list = sorted(df['Asorti'].astype(str).unique())
    inputs['Asorti'] = st.selectbox("Asorti", asorti_list)

    # 6. PASTAL TÜRÜ 
    inputs['Pastal_Turu'] = st.selectbox("Pastal_Turu", sorted(df['Pastal_Turu'].astype(str).unique()))

    # --- DINAMIK PARCA SAYISI HESAPLAMA ---
    if not df_step3.empty and 'Parca_Sayisi' in df_step3.columns:
        # Ortalama değeri alıyoruz
        hesaplanan_parca = float(round(df_step3['Parca_Sayisi'].mean(), 1))
    else:
        hesaplanan_parca = 4.0 

    # SAYISAL GİRİŞLER
    c1, c2 = st.columns(2)
    inputs['Kumas_Eni'] = c1.number_input("Kumas_Eni", 110.0, 200.0, 180.0)
    inputs['Kumas_Gramaji'] = c2.number_input("Kumas_Gramaji", 110.0, 420.0, 150.0)
    
    c3, c4 = st.columns(2)
    inputs['Toplam_Asorti'] = c3.number_input("Toplam_Asorti", 1.0, 50.0, 10.0)
    
    # KUTUNUN GÜNCELLENMESİNİ SAĞLAYAN ANAHTAR: key=f"parca_{secilen_detay}"
    # Bu sayede Model_Detayi her değiştiğinde bu input kutusu kendini resetler.
    inputs['Parca_Sayisi'] = c4.number_input(
        "Parca_Sayisi", 
        1.0, 50.0, 
        value=hesaplanan_parca, 
        step=1.0,
        key=f"parca_{secilen_detay}"
    )

# -----------------------------------------------------------------------------
# 3. HESAPLAMA
# -----------------------------------------------------------------------------
st.divider()

if st.button("HESAPLA", type="primary", use_container_width=True):
    if model:
        try:
            # Girdilerden DataFrame oluştur
            X_new = pd.DataFrame([inputs])
            
            # Modelin beklediği özellik sıralamasını al
            beklenen_siralama = model.feature_names_
            X_new = X_new[beklenen_siralama]

            # Kategorik özellikler listesi
            cat_features = ['Departman', 'Model_Turu', 'Model_Detayi', 'Fit', 'Pastal_Turu', 'Asorti']
            
            X_new_pool = Pool(X_new, cat_features=cat_features)
            
            # Tahmin al
            prediction = model.predict(X_new_pool)[0]
            
            # Sonuç Ekranı
            st.balloons()
            st.markdown(f"""
            <div style="text-align: center; padding: 25px; border: 2px solid #4CAF50; border-radius: 15px; background-color: #f9f9f9;">
                <h3 style="margin: 0; color: #555;">Tahmini Birim Sarfiyat</h3>
                <h1 style="color: #4CAF50; font-size: 60px; margin: 10px 0;">{prediction:.3f} <span style="font-size: 20px;">kg</span></h1>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Bir hata oluştu: {e}")
    else:
        st.error("Model yüklenemedi.")

# Yapmamı istediğin başka bir ekleme var mı?
