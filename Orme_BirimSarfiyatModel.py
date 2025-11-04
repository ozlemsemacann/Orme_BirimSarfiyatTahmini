import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# -----------------------------
# Modeli yükle
# -----------------------------
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("Orme_BirimSarfiyatModel.cbm")
    return model

model = load_model()

# -----------------------------
# Streamlit Arayüzü
# -----------------------------
st.title("🧵 Birim Sarfiyat Tahmini")

st.markdown("Modeli önceden eğittik ve yükledik. Şimdi değerleri gir, tahmini al!")

# Kullanıcıdan girişler
inputs = {}
inputs['Departman'] = st.selectbox("Departman", ['Man','Woman'])
inputs['Model_Turu'] = st.selectbox("Model_Turu", ['ATHLETE','CARDIGAN','JOGGER','KNITTED BOTTOMS','KNITTED SET','KNITTED TOPS','LONG SLEEVE BODY','LONG SLEEVE POLO T-SHIRT','LONG SLEEVE T-SHIRT','SHORT','SHORT SLEEVE POLO T-SHIRT','SHORT SLEEVE T-SHIRT','SWEAT SHIRT','TROUSERS'])
inputs['Fit'] = st.selectbox("Fit", ['BOXY_FIT','OVERSIZE_FIT','REGULAR_FIT','BOXY_FIT','FITTED','RELAX_FIT','SLIM_FIT','LOOSE_FIT','STANDART_FIT','LONG_FIT','CROPPED_FIT','BAGGY_FIT','BOXY_FIT','CLASSIC_FIT','NEW_REGULAR_FIT','BATWING','COMFORT_FIT','OVERSIZE_FIT','OVERSHIRT_FIT','SWEATSHIRT','WIDE_LEG','BAGGY_FIT','REGULAR_FIT','STRAIGHT_FIT','STRAIGHT_FIT','FLARE_FIT','BARREL_FIT','JOGGER_FIT'])
# Kumaş Eni Değerini 130 ile 176 arasına sınırlama
inputs['Kumas_Eni'] = st.number_input(
    "Kumas_Eni",
    min_value=110.0,
    max_value=190.0,
    value=180.0) # Başlangıç değeri
# Kumaş Gramaji 1 ile 30 arasına sınırlama
inputs['Kumas_Gramaji'] = st.number_input(
    "Kumas_Gramaji",
    min_value=110.0,
    max_value=350.0,
    value=150.0) # Başlangıç değeri
inputs['Pastal_Turu'] = st.selectbox("Pastal_Turu", ['Ana_Beden','Yaka','Etek_Kol','Cep','Kol','Kapuson','Etek_Kol_Yaka','Alt','Ust','Kol_Yaka','Etek','Etek_Kol_Yaka_Omuz'])
inputs['Asorti'] = st.selectbox("Asorti", ['XS_S_M_L_XL_XXL_3XL','S_M_L_XL','S_M_L_XL_XXL','XS_S_M_L_XL','XS_S_M_L_XL_XXL','S_M_L_XL_XXL_3XL','XS_S_M_L'])
# Asorti Sayısı Değerini 1 ile 30 arasına sınırlama
inputs['Asorti_Sayisi'] = st.number_input(
    "Asorti_Sayisi",
    min_value=6.0,
    max_value=14.0,
    value=10.0) # Başlangıç değeri
# Parça Sayısı Değerini 1 ile 30 arasına sınırlama
inputs['Parca_Sayisi'] = st.number_input(
    "Parca_Sayisi",
    min_value=1.0,
    max_value=13.0,
    value=4.0) # Başlangıç değeri

# DataFrame oluştur
X_new = pd.DataFrame([inputs])

# Tahmin
if st.button("Tahmin Et"):
    from catboost import Pool

    cat_features = ['Departman', 'Model_Turu', 'Fit',
                    'Pastal_Turu', 'Asorti']

    X_new_pool = Pool(X_new, cat_features=cat_features)
    prediction = model.predict(X_new_pool)[0]
    st.success(f"🔮 Tahmini Birim Sarfiyat: **{prediction:.2f}**")