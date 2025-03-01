import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# กำหนดคอนฟิกของแอพ
st.set_page_config(
    page_title="IS | Nattawut Chongcasee",
    layout="wide",
    initial_sidebar_state="expanded"
)

# สร้างส่วนเนื้อหาหลัก
def main():
    st.title("Intelligent System Project")
    st.markdown("## การวิเคราะห์ข้อมูลด้วย Machine Learning และ Neural Network")
    
    # ข้อมูลผู้จัดทำ
    st.header("ข้อมูลผู้จัดทำ")
    col1, col2 = st.columns([1, 3])
    
    with col1:  
        # ใส่รูปภาพโปรไฟล์ 
        st.image("./images/profile_intell-project.jpg", caption="ผู้จัดทำ")
    
    with col2:
        st.markdown("""
        **รหัสนักศึกษา:** 6604062630200  
        **ชื่อ-นามสกุล:** ณัฐวุฒิ ทรงคาศรี  
        **คณะ/สาขา:** CS  
        **ชั้นปีการศึกษา:** 2  
        **มหาวิทยาลัย:** เทคโนโลยีพระจอมเกล้าพระนครเหนือ(king mongkut's university of technology north bangkok)

        **วิชา:** Intelligent system  
        **ภาคการศึกษา:** 2/2566  
        **อาจารย์ผู้สอน:** ณัฐกิตติ์ จิตรเอื้อตระกูล  
        """)
    
    # รายละเอียดโปรเจค
    st.header("รายละเอียดโปรเจค")
    st.markdown("""
    โปรเจคนี้เป็นส่วนหนึ่งของวิชา IS 2567-2 โดยมีวัตถุประสงค์เพื่อพัฒนาระบบวิเคราะห์ข้อมูลด้วย Machine Learning และ Neural Network 
    โดยใช้ชุดข้อมูล 2 ชุดที่มีความแตกต่างกัน
    
    ### วัตถุประสงค์
    1. เพื่อศึกษาและเรียนรู้เกี่ยวกับการเตรียมข้อมูลสำหรับโมเดล Machine Learning และ Neural Network
    2. เพื่อพัฒนาโมเดล Machine Learning สำหรับการวิเคราะห์และทำนายข้อมูลเพลง
    3. เพื่อพัฒนาโมเดล Neural Network สำหรับการวิเคราะห์และทำนายสภาพอากาศจากรูปภาพ
    4. เพื่อสร้าง Web Application ที่แสดงถึงการทำงานของโมเดลทั้งสอง
    """)
    
    # ชุดข้อมูลที่ใช้
    st.header("ชุดข้อมูลที่ใช้ในโปรเจค")
    
    tab1, tab2 = st.tabs(["ชุดข้อมูลที่ 1 - Spotify Dataset (ML)", "ชุดข้อมูลที่ 2 - Weather Image Dataset (NN)"])
    
    with tab1:
        st.subheader("Spotify Dataset")
        st.markdown("""
        **ที่มา:** Kaggle - Spotify Dataset  
        **ลิงค์:** [Spotify Dataset](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)  
        **ลักษณะข้อมูล:** ข้อมูลคุณลักษณะของเพลงจาก Spotify API  
        **จำนวนข้อมูล:** ประมาณ 232,000 รายการ  
        **จำนวนคุณลักษณะ (features):** 13 คุณลักษณะ  
        **เป้าหมาย:** ทำนายความนิยมของเพลง (popularity score)  
        
        **คุณลักษณะหลักในชุดข้อมูล:**
        - **acousticness:** ระดับความเป็นเพลงอะคูสติก (0.0 ถึง 1.0)
        - **danceability:** ระดับความเหมาะสมในการเต้น (0.0 ถึง 1.0)
        - **energy:** ระดับพลังงานของเพลง (0.0 ถึง 1.0)
        - **instrumentalness:** ความเป็นเพลงบรรเลง ไม่มีเสียงร้อง (0.0 ถึง 1.0)
        - **key:** กุญแจเสียงของเพลง (0-11 ตามทฤษฎีดนตรี)
        - **liveness:** การตรวจจับการแสดงสด (0.0 ถึง 1.0)
        - **loudness:** ความดังโดยเฉลี่ย (dB)
        - **mode:** โหมดของเพลง (0 = minor, 1 = major)
        - **speechiness:** ความเป็นคำพูดในเพลง (0.0 ถึง 1.0)
        - **tempo:** ความเร็วของเพลง (BPM)
        - **time_signature:** จังหวะเพลง (3, 4, 5, ...)
        - **valence:** ความสุข/ความเศร้าของเพลง (0.0 ถึง 1.0)
        - **popularity:** คะแนนความนิยม (0-100) - เป้าหมายในการทำนาย
        """)
        
        # แสดงตัวอย่างของชุดข้อมูล
        try:
            # สร้างข้อมูลตัวอย่าง
            sample_data = {
                'track_name': ['Shape of You', 'Blinding Lights', 'Dance Monkey', 'Watermelon Sugar', 'Levitating'],
                'artist_name': ['Ed Sheeran', 'The Weeknd', 'Tones and I', 'Harry Styles', 'Dua Lipa'],
                'acousticness': [0.581, 0.001, 0.088, 0.122, 0.002],
                'danceability': [0.825, 0.514, 0.824, 0.548, 0.702],
                'energy': [0.652, 0.730, 0.588, 0.816, 0.825],
                'popularity': [98, 96, 93, 88, 85]
            }
            spotify_sample = pd.DataFrame(sample_data)
            st.dataframe(spotify_sample)
        except:
            st.info("ไม่พบไฟล์ข้อมูลตัวอย่าง โปรดอัปโหลดข้อมูลในโฟลเดอร์ data/spotify_tracks.csv")
    
    with tab2:
        st.subheader("Weather Image Dataset")
        st.markdown("""
        **ที่มา:** Kaggle - Multi-class Weather Dataset  
        **ลิงค์:** [Multi-class Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset)  
        **ลักษณะข้อมูล:** ภาพถ่ายสภาพอากาศประเภทต่างๆ  
        **จำนวนข้อมูล:** ประมาณ 1,500 ภาพ  
        **จำนวนคลาส:** 4 ประเภท  
        **เป้าหมาย:** จำแนกประเภทสภาพอากาศจากภาพ  
        
        **ประเภทของสภาพอากาศในชุดข้อมูล:**
        1. **Cloudy** - ท้องฟ้ามีเมฆมาก
        2. **Rainy** - ฝนตก
        3. **Sunny** - แดดจัด
        4. **Sunrise** - พระอาทิตย์ขึ้น
        """)
        
        # แสดงตัวอย่างภาพ
        st.subheader("ตัวอย่างภาพจากชุดข้อมูล")
        col1, col2, col3, col4 = st.columns(4)
        
        # ใช้ภาพตัวอย่างแทนจนกว่าจะมีข้อมูลจริง
        with col1:
            st.image("./images/cloudy_example.jpg", caption="Cloudy")
        with col2:
            st.image("./images/rain_example.jpg", caption="Rainy")
        with col3:
            st.image("./images/shine_example.jpg", caption="Sunny")
        with col4:
            st.image("./images/sunrise_example.jpg", caption="Sunrise")
    
    # เทคโนโลยีที่ใช้
    st.header("เทคโนโลยีที่ใช้ในการพัฒนา")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### Python")
        st.markdown("ภาษาหลักในการพัฒนา")
    
    with col2:
        st.markdown("### Streamlit")
        st.markdown("Framework สำหรับพัฒนา Web Application")
    
    with col3:
        st.markdown("### Scikit-learn")
        st.markdown("Library สำหรับพัฒนาโมเดล Machine Learning")
    
    with col4:
        st.markdown("### TensorFlow/Keras")
        st.markdown("Library สำหรับพัฒนาโมเดล Neural Network และประมวลผลภาพ")
    
    # คำแนะนำการใช้งาน
    st.header("คำแนะนำการใช้งานเว็บแอป")
    st.info("""
    สามารถเลือกหน้าที่ต้องการจาก Sidebar ด้านซ้ายเพื่อเข้าสู่ส่วนต่างๆ ของโปรเจค:
    
    - **Machine Learning Description** - อธิบายการเตรียมข้อมูล ทฤษฎีและแนวคิดเกี่ยวกับ Machine Learning
    - **Neural Network Description** - อธิบายการเตรียมข้อมูล ทฤษฎีและแนวคิดเกี่ยวกับ Neural Network
    - **Machine Learning Demo** - ทดลองใช้งานโมเดล Machine Learning สำหรับวิเคราะห์และทำนายข้อมูลเพลง
    - **Neural Network Demo** - ทดลองใช้งานโมเดล Neural Network สำหรับจำแนกสภาพอากาศจากรูปภาพ
    """)

if __name__ == "__main__":
    main()