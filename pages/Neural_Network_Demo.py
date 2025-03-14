import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io

# Setup หน้าเพจ
st.set_page_config(page_title="Weather Classifier", page_icon="🌈", layout="wide")

# หัวข้อและคำอธิบาย
st.title("✨ Weather Image Classification By Neural Network ✨")
st.write("""
### ลองเล่น Demo จำแนกรูปสภาพอากาศกันเถอะ! 

Upload รูปสภาพอากาศที่คุณอยากให้ classify ได้เลย! 
โมเดล Neural Network ของผมจะจำแนกรูปเป็นหนึ่งใน category ต่อไปนี้:
""")

# แสดง categories ในรูปแบบที่น่าสนใจ
col_cat1, col_cat2, col_cat3, col_cat4 = st.columns(4)
with col_cat1:
    st.info("☁️ **Cloudy Mood**")
with col_cat2:
    st.info("🌧️ **Rainy Vibes**")
with col_cat3:
    st.info("☀️ **Sunny Day**")
with col_cat4:
    st.info("🌅 **Sunrise Magic**")

# โหลดโมเดล
@st.cache_resource
def load_nn_model():
    try:
        # Try โหลด fine-tuned model ก่อน
        model_path = os.path.join('models', 'weather_cnn_model_finetuned.h5')
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            # Fallback ไปใช้โมเดลปกติ
            model_path = os.path.join('models', 'weather_cnn_model.h5')
            model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Oops! มีปัญหาตอนโหลดโมเดล: {e}")
        return None

# Preprocessing รูปภาพ
def preprocess_image(img):
    # Resize รูป
    img = img.resize((224, 224))
    # Convert เป็น array และ normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ฟังก์ชัน predict
def predict_weather(img_array, model):
    # Class names แบบ fun & friendly
    class_names = ['Cloudy Mood ☁️', 'Rainy Vibes 🌧️', 'Sunny Day ☀️', 'Sunrise Magic 🌅']
    
    # ทำนาย
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100
    
    return predicted_class, confidence, prediction[0]

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Upload Your Weather Snap!")
    uploaded_file = st.file_uploader("Choose a cool weather pic...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # แสดงรูป
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption='Your awesome pic! 📸', use_column_width=True)
            
            # Preprocess
            img_array = preprocess_image(img)
            
            # Predict เมื่อกดปุ่ม
            if st.button('✨ Classify My Weather! ✨'):
                with st.spinner('รอแป๊ปนะ ผมขอเวลาคิดสักครู่... 🤔'):
                    # โหลดโมเดล
                    model = load_nn_model()
                    
                    if model:
                        predicted_class, confidence, prediction_scores = predict_weather(img_array, model)
                        
                        # แสดงผล prediction แบบสนุกๆ
                        st.success(f"### It's {predicted_class}! ✅")
                        
                        # ข้อความตามความมั่นใจ
                        if confidence > 90:
                            st.info(f"มั่นใจมากๆ มั่นใจที่สุดในโลก! ({confidence:.1f}%) 🚀")
                        elif confidence > 70:
                            st.info(f"ค่อนข้างมั่นใจนะ ({confidence:.1f}%) 👍")
                        else:
                            st.info(f"ไม่ค่อยแน่ใจเท่าไหร่เลย... ({confidence:.1f}%) 🤔")
                        
                        # Chart สวยๆ
                        class_names = ['Cloudy Mood ☁️', 'Rainy Vibes 🌧️', 'Sunny Day ☀️', 'Sunrise Magic 🌅']
                        fig, ax = plt.subplots(figsize=(8, 4))
                        y_pos = np.arange(len(class_names))
                        bars = ax.barh(y_pos, prediction_scores*100, align='center', 
                                 color=['cornflowerblue', 'royalblue', 'gold', 'orange'])
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(class_names)
                        ax.invert_yaxis()  # Labels read top-to-bottom
                        ax.set_xlabel('Confidence %')
                        ax.set_title('AI Prediction Breakdown')
                        
                        # แสดง chart
                        st.pyplot(fig)
        except Exception as e:
            st.error(f"Oops! มีปัญหาเกิดขึ้น: {e}")

with col2:
    st.subheader("🧠 Behind the Magic")
    
    # สไลด์แสดงข้อมูลโมเดลแบบเป็นมิตร
    st.write("### 🤖 AI Model")
    st.write("""
    เราใช้ **MobileNetV2** เป็น backbone ของโมเดล - เป็น CNN สุดตึงที่ทั้งเร็วและแม่นยำ!
    
    **Architecture highlights:**
    - Transfer learning จาก model ที่ pre-train บน ImageNet มาแล้ว
    - Fine-tune ด้วย dataset รูปสภาพอากาศจริงๆ
    - Dropout layers เพื่อป้องกันการ overfitting 
    - Global Average Pooling เพื่อลดจำนวน parameters
    """)
    
    # พยายามโหลดและแสดง metrics
    try:
        if os.path.exists('data/training_history.png'):
            st.write("### 📊 Training Stats")
            st.image('data/training_history.png', use_column_width=True)
        
        if os.path.exists('data/confusion_matrix.png'):
            st.write("### 🧩 Confusion Matrix")
            st.image('data/confusion_matrix.png', use_column_width=True)
    except Exception as e:
        st.warning("Performance metrics ยังไม่พร้อมแสดงผล (training data needed)")
    
    # คำแนะนำใช้งาน
    st.write("### 🚀 How to Use")
    st.write("""
    **ง่ายนิดเดียว**
    1. Upload รูปสภาพอากาศที่คุณถ่ายหรือมีอยู่แล้ว
    2. กดปุ่ม "Classify My Weather!"
    3. รอแป๊บนึง แล้วดูผลทำนาย
    4. Bar chart จะแสดงว่าโมเดลนี้คิดว่ารูปของคุณเป็นสภาพอากาศแบบไหนบ้าง
    """)

# ตัวอย่างภาพ
st.subheader("📸 ตัวอย่างภาพ (Sample Pics)")
st.write("ไม่มีรูปจะลอง? ลองดูตัวอย่างสภาพอากาศแต่ละแบบได้ที่นี่:")

# สร้างเลย์เอาต์สำหรับภาพตัวอย่าง
sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

# ตัวอย่างภาพพร้อมคำอธิบายที่สนุกขึ้น
with sample_col1:
    st.write("☁️ **Cloudy Mood**")
    st.image("./images/cloudy_example2.jpg")
    st.caption("อากาศดีจังเลย")

with sample_col2:
    st.write("🌧️ **Rainy Vibes**")
    st.image("./images/rain_example2.jpg")
    st.caption("เวลาที่เหมาะกับการนอน และเพลงเพราะๆ")

with sample_col3:
    st.write("☀️ **Sunny Day**")
    st.image("./images/shine_example2.jpg")
    st.caption("แดดจ้า! Perfect day for outdoor!")

with sample_col4:
    st.write("🌅 **Sunrise Magic**")
    st.image("./images/sunrise_example2.jpg")
    st.caption("ช่วงเวลาแห่งการเริ่มต้น")

# Tech details แบบขยายได้
with st.expander("🔬 Tech Geek Corner"):
    st.write("""
    ### 🤓 รายละเอียดทางเทคนิค
    
    **Model specs:**
    - **Base**: MobileNetV2 (pre-trained on ImageNet dataset)
    - **Input**: 224x224x3 (RGB images)
    - **Strategy**: Transfer learning + fine-tuning
    - **Optimizer**: Adam with learning rate scheduling
    - **Regularization**: Dropout + data augmentation
    
    **Image preprocessing pipeline:**
    1. Resize ภาพให้เป็น 224x224 pixels
    2. Normalize pixel values ให้อยู่ในช่วง 0-1
    3. ตอน training มีการใช้ data augmentation เพิ่มความหลากหลาย
    
    **Performance optimization:**
    - โมเดลนี้ balance ระหว่าง accuracy กับ speed ได้ดี
    - Fine-tuning ช่วยเพิ่ม performance บน specific weather conditions
    - Confidence scores บอกความน่าเชื่อถือของแต่ละ prediction
    """)

# Footer ที่ดูทันสมัย
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>✨ 🌦️ Predicting weather | Created by Nattawut Chongcasee ✨</p>
</div>
""", unsafe_allow_html=True)