import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# กำหนดคอนฟิกของหน้า
st.set_page_config(
    page_title="ML Demo - Spotify Popularity Prediction",
    page_icon="🎵",
    layout="wide"
)

# โหลดโมเดลที่ฝึกฝนไว้แล้ว
@st.cache_resource
def load_models():
    models = {}
    try:
        models['Linear Regression'] = pickle.load(open('models/spotify_popularity_lr.pkl', 'rb'))
        models['Random Forest'] = pickle.load(open('models/spotify_popularity_rf.pkl', 'rb'))
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    return models

# ฟังก์ชันทำนายความนิยมของเพลง
def predict_popularity(models, features):
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(features)[0]
    return predictions

# โหลดโมเดล
models = load_models()

st.title("🎵 ทำนายความนิยมของเพลง (Spotify Popularity Prediction)")
st.markdown("""
เปรียบเทียบผลการทำนายระหว่าง **Linear Regression** และ **Random Forest Regressor**
""")

def generate_random_input():
    return {
        'acousticness': np.random.uniform(0, 1),
        'danceability': np.random.uniform(0, 1),
        'energy': np.random.uniform(0, 1),
        'instrumentalness': np.random.uniform(0, 1),
        'key': np.random.randint(0, 12),
        'liveness': np.random.uniform(0, 1),
        'loudness': np.random.uniform(-60, 0),
        'mode': np.random.choice([0, 1]),
        'speechiness': np.random.uniform(0, 1),
        'tempo': np.random.uniform(50, 200),
        'time_signature': np.random.choice([3, 4, 5, 6, 7]),
        'valence': np.random.uniform(0, 1)
    }

# อินพุตจากผู้ใช้
st.header("ป้อนข้อมูลคุณลักษณะของเพลง")
col1, col2, col3 = st.columns(3)

if st.button("🎲 สุ่มค่าอินพุต"):
    random_values = generate_random_input()
else:
    random_values = None

with col1:
    track_name = st.text_input("ชื่อเพลง", "Shape of You")
    artist_name = st.text_input("ชื่อศิลปิน", "Ed Sheeran")
    acousticness = st.slider("Acousticness", 0.0, 1.0, random_values['acousticness'] if random_values else 0.5, 0.01)
    danceability = st.slider("Danceability", 0.0, 1.0, random_values['danceability'] if random_values else 0.5, 0.01)

with col2:
    energy = st.slider("Energy", 0.0, 1.0, random_values['energy'] if random_values else 0.5, 0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, random_values['instrumentalness'] if random_values else 0.0, 0.01)
    key = st.selectbox("Key", list(range(12)), random_values['key'] if random_values else 0)
    liveness = st.slider("Liveness", 0.0, 1.0, random_values['liveness'] if random_values else 0.1, 0.01)

with col3:
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, random_values['loudness'] if random_values else -10.0, 0.1)
    mode = st.selectbox("Mode", [0, 1], index=[0, 1].index(random_values['mode']) if random_values else 1, format_func=lambda x: "Minor" if x == 0 else "Major")
    speechiness = st.slider("Speechiness", 0.0, 1.0, random_values['speechiness'] if random_values else 0.1, 0.01)
    tempo = st.slider("Tempo (BPM)", 50.0, 200.0, random_values['tempo'] if random_values else 120.0, 0.1)
    time_signature = st.selectbox("Time Signature", [3, 4, 5, 6, 7], index=[3, 4, 5, 6, 7].index(random_values['time_signature']) if random_values else 1)
    valence = st.slider("Valence", 0.0, 1.0, random_values['valence'] if random_values else 0.5, 0.01)

if st.button("ทำนายความนิยม", use_container_width=True):
    if models:
        features = pd.DataFrame({
            'acousticness': [acousticness],
            'danceability': [danceability],
            'energy': [energy],
            'instrumentalness': [instrumentalness],
            'key': [key],
            'liveness': [liveness],
            'loudness': [loudness],
            'mode': [mode],
            'speechiness': [speechiness],
            'tempo': [tempo],
            'time_signature': [time_signature],
            'valence': [valence]
        })
        
        predictions = predict_popularity(models, features)
        
        # แสดงผลลัพธ์
        st.success(f"**ผลการทำนายของ {track_name} - {artist_name}:**")
        for model_name, pred in predictions.items():
            st.markdown(f"- **{model_name}:** {pred:.1f}/100")
        
        # แสดงกราฟเปรียบเทียบ
        st.subheader("📊 การเปรียบเทียบโมเดล")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(predictions.keys(), predictions.values(), color=['blue', 'green'])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Popularity Score")
        st.pyplot(fig)



# แสดงข้อมูลเกี่ยวกับโมเดลที่ใช้
st.markdown("""### ความหมายของคุณลักษณะ
- **acousticness**: ระดับความเป็นเพลงอะคูสติก (0.0-1.0)
- **danceability**: ระดับความเหมาะสมในการเต้น (0.0-1.0) 
- **energy**: ระดับพลังงานของเพลง (0.0-1.0)
- **instrumentalness**: ความเป็นเพลงบรรเลง ไม่มีเสียงร้อง (0.0-1.0)
- **key**: กุญแจเสียงของเพลง (0-11)
- **liveness**: การตรวจจับการแสดงสด (0.0-1.0)
- **loudness**: ความดังโดยเฉลี่ย (dB)
- **mode**: โหมดของเพลง (0 = minor, 1 = major)
- **speechiness**: ความเป็นคำพูดในเพลง (0.0-1.0)
- **tempo**: ความเร็วของเพลง (BPM)
- **time_signature**: จังหวะเพลง (3, 4, 5, ...)
- **valence**: ความสุข/ความเศร้าของเพลง (0.0-1.0)
""")