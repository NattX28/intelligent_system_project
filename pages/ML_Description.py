import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    # ตั้งค่าหน้าเพจ
    st.set_page_config(
        page_title="Spotify Song Popularity Prediction",
        page_icon="🎵",
        layout="wide"
    )
    
    # CSS custom
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1DB954;  /* Spotify Green */
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #191414;  /* Spotify Black */
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1DB954;
    }
    .highlight {
        background-color: #f0f0f0;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1DB954;
    }
    .caption {
        font-size: 0.9rem;
        color: #666;
        text-align: center;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">การวิเคราะห์และทำนายความนิยมของเพลง Spotify</div>', unsafe_allow_html=True)
    
    # Introduction with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("""
        จุดเริ่มต้นเกิดจากที่ผมสนใน dataset อะไรก็ได้ที่เกี่ยวกับเพลง ผมจึงได้ให้ claude ai ค้นหา dataset เกี่ยวกับเพลงที่มีความไม่สมบูรณ์จาก Kaggleให้ จนได้ข้อมูล [Spotify Dataset](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) มา ซึ่งเป็นข้อมูลคุณลักษณะของเพลงจาก Spotify API ซึ่งผมต้องการจะทำนายความโด่งดังของเพลง
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Placeholder for logo or image
        st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png", 
                 width=250)
        

    # Features card
    st.markdown('<div class="sub-header">คุณลักษณะหลักในชุดข้อมูล</div>', unsafe_allow_html=True)
    
    # สร้างรูปแบบการแสดงผล 3 คอลัมน์สำหรับ features
    col1, col2, col3 = st.columns(3)
    
    features = [
        {"name": "acousticness", "desc": "ระดับความเป็นเพลงอะคูสติก (0.0 ถึง 1.0)"},
        {"name": "danceability", "desc": "ระดับความเหมาะสมในการเต้น (0.0 ถึง 1.0)"},
        {"name": "energy", "desc": "ระดับพลังงานของเพลง (0.0 ถึง 1.0)"},
        {"name": "instrumentalness", "desc": "ความเป็นเพลงบรรเลง ไม่มีเสียงร้อง (0.0 ถึง 1.0)"},
        {"name": "key", "desc": "กุญแจเสียงของเพลง (0-11 ตามทฤษฎีดนตรี)"},
        {"name": "liveness", "desc": "การตรวจจับการแสดงสด (0.0 ถึง 1.0)"},
        {"name": "loudness", "desc": "ความดังโดยเฉลี่ย (dB)"},
        {"name": "mode", "desc": "โหมดของเพลง (0 = minor, 1 = major)"},
        {"name": "speechiness", "desc": "ความเป็นคำพูดในเพลง (0.0 ถึง 1.0)"},
        {"name": "tempo", "desc": "ความเร็วของเพลง (BPM)"},
        {"name": "time_signature", "desc": "จังหวะเพลง (3, 4, 5, ...)"},
        {"name": "valence", "desc": "ความสุข/ความเศร้าของเพลง (0.0 ถึง 1.0)"},
        {"name": "popularity", "desc": "คะแนนความนิยม (0-100) - เป้าหมายในการทำนาย"}
    ]
    
    # แบ่งคุณลักษณะเป็น 3 กลุ่มสำหรับการแสดงผลในคอลัมน์
    features_columns = [features[i:i+5] for i in range(0, len(features), 5)]
    
    with col1:
        for feature in features_columns[0]:
            st.markdown(f"**{feature['name']}**: {feature['desc']}")
    
    with col2:
        for feature in features_columns[1]:
            st.markdown(f"**{feature['name']}**: {feature['desc']}")
    
    with col3:
        if len(features_columns) > 2:
            for feature in features_columns[2]:
                st.markdown(f"**{feature['name']}**: {feature['desc']}")
    
    # ส่วนแสดงผลการวิเคราะห์ข้อมูล
    st.markdown('<div class="sub-header">การวิเคราะห์และการเตรียมข้อมูล</div>', unsafe_allow_html=True)
    
    # สร้าง tab สำหรับแต่ละขั้นตอนการวิเคราะห์
    tabs = st.tabs([
        "📊 Head & Tail", 
        "🔍 Null Check", 
        "📉 Outliers", 
        "🧹 Data Cleaning", 
        "⚖️ Scaling", 
        "📈 Linear Regression", 
        "🌲 Random Forest", 
        "📏 Model Evaluation", 
        "📊 Histogram"
    ])
    
    with tabs[0]:
        st.subheader("ผมเริ่มจากดู head ของข้อมูลก่อน")
        st.image("./images/ML_Desc1.png")
        st.write("จากนั้นดู tail ของข้อมูล")
        st.image("./images/ML_Desc2.png")
    
    with tabs[1]:
        st.subheader("และผมก็ได้ทำการตรวจสอบก่อนว่าข้อมูลมี null ไหม ถ้ามีแล้วมันมีกี่อัน")
        st.image("./images/ML_Desc3.png")
    
    with tabs[2]:
        st.subheader("หลังจากนั้นตรวจหา outlier ของข้อมูลด้วยการใช้แผนภาพกล่องเพื่อแสดงการกระจายตัวของคุณลักษณะเสียงจาก Spotify ทั้ง 12 ฟีเจอร์ พบว่ามีหลายอันเลย")
        st.image("./images/ML_Desc4.png")
        
    
    with tabs[3]:
        st.subheader("ผมเลยทำการ clean data โดยมีการตรวจสอบและลบ Missing Values กำหนดตัวแปร features เก็บชื่อคอลัมน์ที่เป็นคุณลักษณะ (12 คอลัมน์) กำหนดตัวแปร target เก็บชื่อคอลัมน์เป้าหมาย (popularity)")
        st.image("./images/ML_Desc5.png")
    
    with tabs[4]:
        st.subheader("ต่อมาทำการจัดการ Outliers โดยกรองข้อมูลโดยเก็บเฉพาะแถวที่มีค่าอยู่ระหว่างขอบเขตล่างและขอบเขตบน")
        st.image("./images/ML_Desc6.png")
        st.subheader("step ต่อมาก็ได้ทำการแยกข้อมูล")
        st.image("./images/ML_Desc7.png")
        st.subheader("สร้าง StandardScaler เพื่อปรับข้อมูลให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1 เพราะต้องการเปรียบเทียบ Features ที่มีการกระจายตัวแตกต่างกัน และทำการ fit และ transform ข้อมูลชุดเทรน transform ข้อมูลชุดทดสอบด้วย scaler เดียวกัน")
        st.image("./images/ML_Desc8.png")
    
    with tabs[5]:
        st.subheader("สร้างโมเดล Linear Regression ตามด้วยเทรนโมเดลด้วยข้อมูลที่ผ่านการ scale แล้ว จากนั้นทำนายค่า popularity ของข้อมูลทดสอบ")
        st.image("./images/ML_Desc9.png")
    
    with tabs[6]:
        st.subheader("สร้างโมเดล Random Forest Regressor ที่มี 15 ต้นไม้ เพราะถ้ามากกว่านี้ผมคิดว่าจำนวนต้นไม้เริ่มไม่มีผลแล้ว และผมต้องการลดขนาดของโมเดลด้วย จากนั้นเทรนโมเดลด้วยข้อมูลเทรน (สังเกตว่าใช้ข้อมูลที่ไม่ผ่านการ scale)และทำนายค่า popularity ของข้อมูลทดสอบ")
        st.image("./images/ML_Desc10.png")
    
    with tabs[7]:
        st.subheader("จากนั้นผมประเมินโมเดล Linear Regression และ  Random Forest ด้วย MAE (Mean Absolute Error) และ RMSE (Root Mean Squared Error) ซึ่งอยุ่ในจุดที่ผมยอมรับได้")
        st.image("./images/ML_Desc11.png")
        
        # แสดงตารางผลลัพธ์เปรียบเทียบ (ตัวอย่าง)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("💡 **ค่า MAE และ RMSE ยิ่งน้อยยิ่งดี** \n\nโมเดล Random Forest ให้ค่าความคลาดเคลื่อนที่ต่ำกว่า Linear Regression ทั้งในแง่ของ MAE และ RMSE")
        
        with col2:
            # สร้างข้อมูลจำลองสำหรับตาราง
            results = {
                "Model": ["Linear Regression", "Random Forest"],
                "MAE": ["12.93", "8.87"],
                "RMSE": ["16.28", "12.76"]
            }
            st.table(pd.DataFrame(results))
    
    with tabs[8]:
        st.subheader("ลองสร้างกราฟ histogram เปรียบเทียบผลลัพธ์ดู โดยแสดงการกระจายตัวของค่า popularity จริง (สีฟ้า) และค่าที่ทำนายได้จาก Random Forest (สีแดง)")
        st.image("./images/ML_Desc12.png")
    
    # สรุปผลและข้อเสนอแนะ
    st.markdown('<div class="sub-header">สรุปผลการวิเคราะห์</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ผลลัพธ์ที่ได้
        จากการวิเคราะห์และทดสอบโมเดล พบว่า:
        
        1. โมเดล Random Forest ให้ผลการทำนายที่แม่นยำกว่าโมเดล Linear Regression
        2. คุณลักษณะที่มีผลต่อความนิยมของเพลงมีความสัมพันธ์แบบไม่เชิงเส้น
        3. สามารถใช้คุณลักษณะทางเสียงเพื่อทำนายความนิยมของเพลงได้ในระดับที่น่าพอใจ
        """)
    
    with col2:
        st.markdown("""
        ### แนวทางการพัฒนาในอนาคต
        
        1. เพิ่มข้อมูลอื่นๆ เช่น ข้อมูลศิลปิน, เนื้อเพลง, หรือข้อมูลจาก social media
        2. ปรับปรุงการจัดการ outliers ด้วยเทคนิคอื่นๆ
        3. ทดลองใช้โมเดลที่ซับซ้อนขึ้น เช่น Neural Networks
        4. แบ่งการวิเคราะห์ตามแนวเพลง (Genre) เพื่อเพิ่มความแม่นยำ
        """)
    
    # Footer
    st.markdown("""---""")
    st.markdown("""
    <div class="caption">
    Predicting Spotify Song Popularity | Created by Nattawut Chongcasee
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    app()