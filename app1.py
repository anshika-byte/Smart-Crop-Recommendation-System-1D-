# =========================
# 1. IMPORT LIBRARIES
# =========================
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# =========================
# 2. LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")

    crop_labels = df['label'].astype('category').cat.categories

    df.rename(columns={
        'temperature': 'temp',
        'label': 'crop'
    }, inplace=True)

    df['soil_type'] = pd.cut(df['N'], bins=3, labels=['sandy','loamy','clay'])

    df = df[['N','P','K','temp','humidity','rainfall','soil_type','crop']]

    df['soil_type'] = df['soil_type'].astype('category').cat.codes
    df['crop'] = df['crop'].astype('category').cat.codes

    return df, crop_labels

# =========================
# 3. TRAIN MODEL
# =========================
@st.cache_resource
def train_model(df):
    X = df.drop('crop', axis=1)
    y = df['crop']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train_sm, y_train_sm)

    return model, scaler

# Load
df, crop_labels = load_data()
model, scaler = train_model(df)

# =========================
# 4. BACKGROUND + STYLING
# =========================
page_bg = """
<style> [data-testid="stAppViewContainer"] { background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef"); background-size: cover; background-position: center; }

/* Dark overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.55);
    z-index: -1;
}
.result-box {
    background: linear-gradient(135deg, #ff1744, #ff5252);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    box-shadow: 0px 0px 20px rgba(255, 0, 0, 0.6);
    margin-top: 20px;
}
/* Bold white text */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: white !important;
    font-weight: bold !important;
}

/* Button style */
.stButton>button {
    font-weight: bold;
    background-color: #28a745;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
}

/* Slider text */
.stSlider label {
    font-weight: bold;
    color: white;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# 5. TITLE
# =========================
st.markdown(
    "<h1 style='text-align: center; font-weight: 900; text-shadow: 2px 2px 10px black;'>🌱 Smart Crop Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Enter soil nutrients and environmental conditions</p>",
    unsafe_allow_html=True
)

# =========================
# 6. INPUT UI
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌿 Soil Nutrients")
    N = st.slider("Nitrogen (N)", 0, 150, 50)
    P = st.slider("Phosphorus (P)", 0, 150, 50)
    K = st.slider("Potassium (K)", 0, 150, 50)

with col2:
    st.subheader("🌦 Environment")
    temp = st.slider("Temperature (°C)", 10, 45, 25)
    humidity = st.slider("Humidity (%)", 10, 100, 60)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

st.subheader("🪨 Soil Type")
soil = st.selectbox("", ['sandy','loamy','clay'])

soil_map = {'sandy':0, 'loamy':1, 'clay':2}

# =========================
# 7. PREDICTION
# =========================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Crop", use_container_width=True):

    input_data = np.array([[N, P, K, temp, humidity, rainfall, soil_map[soil]]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]

    crop_name = crop_labels[pred]

    st.markdown(
    f"""
    <div class="result-box">
        🌾 Recommended Crop: {str(crop_name).upper()}
    </div>
    """,
    unsafe_allow_html=True
)

