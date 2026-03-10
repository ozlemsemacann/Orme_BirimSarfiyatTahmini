import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os

# -----------------------------------------------------------------------------
# 1. AYARLAR VE DOSYA YÖNETİMİ
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Örme Sarfiyat Tahmini", layout="wide")

# Dosya yollarını dinamik bul
current_dir = os.path.dirname(os.path.abspath(__file__))

# DİKKAT: Excel dosyanın adını buradakiyle aynı yapmalısın
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

# Veri ve Modeli Yükle
df = load_data()
model = load_model()

# Eğer veri yoksa durdur
if df is None or model is None:
    st.stop()

# -----------------------------------------------------------------------------
# 2. BASAMAKLI FİLTRELEME (CASCADING FILTERS)
# -----------------------------------------------------------------------------
st.title("🧶 Örme Birim Sarfiyat Tahmini")
st.success(f"✅ Modeli önceden eğittik ve yükledik. Şimdi değerleri gir, tahmini al!")

inputs = {}
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Model Seçimi")

    # 1. DEPARTMAN
    dept_list = sorted(df['Departman'].astype(str).unique())
    secilen_dept = st.selectbox("Departman", dept_list)
    inputs['Departman'] = secilen_dept
    
    # FİLTRE 1: Departmana göre daralt
    df_step1 = df[df['Departman'] == secilen_dept]

    # 2. MODEL TÜRÜ
    tur_list = sorted(df_step1['Model_Turu'].astype(str).unique())
    secilen_tur = st.selectbox("Model_Turu", tur_list)
    inputs['Model_Turu'] = secilen_tur
    
    # FİLTRE 2: Türe göre daralt
    df_step2 = df_step1[df_step1['Model_Turu'] == secilen_tur]

    # 3. MODEL DETAYI (YENİ EKLENEN ADIM)
    # Filtrelenmiş 2. adımdan Model Detaylarını getiriyoruz
    detay_list = sorted(df_step2['Model_Detayi'].astype(str).unique())
    secilen_detay = st.selectbox("Model_Detayi", detay_list)
    inputs['Model_Detayi'] = secilen_detay
    
    # FİLTRE 3: Model Detayına göre daralt
    df_step3 = df_step2[df_step2['Model_Detayi'] == secilen_detay]

    # 4. FIT
    fit_list = sorted(df_step3['Fit'].astype(str).unique())
    secilen_fit = st.selectbox("Fit", fit_list)
    inputs['Fit'] = secilen_fit

    # FİLTRE 4: Fit'e göre daralt (Asorti için hazırlık)
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

    # SAYISAL GİRİŞLER
    c1, c2 = st.columns(2)
    inputs['Kumas_Eni'] = c1.number_input("Kumas_Eni", 110.0, 200.0, 180.0)
    inputs['Kumas_Gramaji'] = c2.number_input("Kumas_Gramaji", 110.0, 420.0, 150.0)
    
    c3, c4 = st.columns(2)
    inputs['Toplam_Asorti'] = c3.number_input("Toplam_Asorti", 6.0, 14.0, 10.0)
    inputs['Parca_Sayisi'] = c4.number_input("Parca_Sayisi", 1.0, 13.0, 4.0)

# -----------------------------------------------------------------------------
# 3. HESAPLAMA
# -----------------------------------------------------------------------------
st.divider()

if st.button("HESAPLA", type="primary", use_container_width=True):
    if model:
        try:
            # Girdilerden DataFrame oluştur
            X_new = pd.DataFrame([inputs])
            
            # --- OTOMATİK SIRALAMA ---
            beklenen_siralama = model.feature_names_
            X_new = X_new[beklenen_siralama]

            # Kategorik özellikler listesine Model_Detayi EKLENDİ
            cat_features = ['Departman', 'Model_Turu', 'Model_Detayi', 'Fit', 'Pastal_Turu', 'Asorti']
            
            X_new_pool = Pool(X_new, cat_features=cat_features)
            
            # Numpy Array format hatasını çözen indeksleme
            prediction = model.predict(X_new_pool)[0]
            
            st.success(f"🧶 Tahmini Birim Sarfiyat: **{prediction:.3f} kg**")
            
        except KeyError as e:
            st.error(f"Sütun Hatası: Model {e} isimli bir veri bekliyor ama kodda bu isim eksik veya yanlış yazılmış.")
        except Exception as e:
            st.error(f"Hesaplama Hatası: {e}")
            st.info("İpucu: Excel dosyasındaki sütun isimlerinin harf büyüklüklerinin kodla ('Model_Detayi' gibi) aynı olduğundan emin olun.")
    else:
        st.error("Model yüklenemediği için hesaplama yapılamıyor.")
