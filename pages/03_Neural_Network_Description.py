import streamlit as st

def main():
    # Set page config
    st.set_page_config(
        page_title="การทำนายสภาพอากาศจากรูปภาพ",
        page_icon="🌦️",
        layout="wide"
    )

    # Header
    st.title("🌦️ การทำนายสภาพอากาศจากรูปภาพด้วย Neural Network")
    
    # Introduction
    st.markdown("""
    <div style="background-color:#f0f5ff;padding:20px;border-radius:10px;margin-bottom:20px">
    <h3 style="color:#1E88E5">จุดเริ่มต้น</h3>
    
    ผมมีความสนใจที่อยากจะทำเว็บที่เราสามารถอัปโหลดรูปภาพเพื่อทำนายสภาพอากาศได้ โดยใช้ Neural network เอาล่ะผมเลยเริ่มหา dataset ใน kaggle ที่ผมใช้ใน machine learning model ก่อนหน้าที่ผมได้ทำไป นั่งหาอยู่นาน แต่ยังไม่เจออันที่ตรงใจ เลยพึ่งพา claude ai ตัวเก่งของผม เพื่อหา dataset ที่มีความไม่สมบูรณ์ จน cluade ai แนะนำ [Multi-class Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset) จาก kaggle(อีกแล้ว)ให้ ผมเลยโอเค! จัดอันนี้เลยแล้วกัน
    </div>
    """, unsafe_allow_html=True)

    # Understanding the dataset
    with st.container():
        st.header("📊 ทำความเข้าใจกับ Dataset")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ก่อนอื่นเลย มาทำความเข้าใจกับ dataset ก่อนดีกว่า
            
            - **ลักษณะข้อมูล**: ภาพถ่ายสภาพอากาศประเภทต่างๆ
            - **จำนวนข้อมูล**: ประมาณ 1,500 ภาพ
            - **จำนวนคลาส**: 4 ประเภท
            - **เป้าหมาย**: จำแนกประเภทสภาพอากาศจากภาพ
            
            **ประเภทของสภาพอากาศในชุดข้อมูล**:
            - **Cloudy** - ท้องฟ้ามีเมฆมาก
            - **Rainy** - ฝนตก
            - **Sunny** - แดดจัด
            - **Sunrise** - พระอาทิตย์ขึ้น
            """)
        
        with col2:
            st.image("./images/rain_example3.jpg", caption="ตัวอย่างภาพจาก Dataset")
    
    st.markdown("---")
    st.markdown("<h3 style='text-align:center'>เอาล่ะ มาเริ่มกันเลยยยยย!</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Libraries
    with st.container():
        st.header("📚 Import ไลบรารีที่จำเป็น")
        
        st.markdown("""
        ก่อนอื่น เราต้อง import ไลบรารีที่จำเป็นก่อน มีหลายตัวเลย เช่น
        - **TensorFlow และ Keras** → เอาไว้สร้างโมเดล (ขาดไม่ได้เลย)
        - **Scikit-learn** → ใช้แบ่งข้อมูลและประเมินผล
        - **Matplotlib และ Seaborn** → สำหรับวิเคราะห์และแสดงผลข้อมูล
        """)
        
        st.image("./images/NN_Desc1.png", caption="Import Libraries")
    
    # Initial Setup
    with st.container():
        st.header("⚙️ ตั้งค่าเริ่มต้น")
        
        st.markdown("""
        เราจะกำหนดค่า random seed ไว้ก่อน เพื่อให้ได้ผลลัพธ์ที่สามารถทำซ้ำได้ทุกครั้ง
        จากนั้น กำหนดค่าพารามิเตอร์หลักที่ใช้ในโมเดล
        - **IMG_SIZE**: กำหนดขนาดภาพที่ใช้เทรน (224x224 pixels)
        - **BATCH_SIZE**: จำนวนภาพที่ประมวลผลพร้อมกันในแต่ละรอบ (32 รูป)
        - **EPOCHS**: จำนวนรอบในการเทรนโมเดล (20 รอบกำลังดี)
        - **NUM_CLASSES**: จำนวนคลาสที่เราจะจำแนก (4 ประเภท นั่นคือ Cloudy, Rainy, Sunny และ Sunrise)
        """)
        
        st.image("./images/NN_Desc2.png", caption="การตั้งค่าพารามิเตอร์")
    
    # Load Data
    with st.container():
        st.header("📥 โหลดข้อมูล")
        
        st.markdown("""
        ข้อมูลภาพที่ใช้เป็นภาพเกี่ยวกับสภาพอากาศ ซึ่งอยู่บน Google Drive ที่ผมไปอัปโหลดไว้บน drive ของผม เราจะดาวน์โหลดและแตกไฟล์ออกมา
        """)
        
        st.image("./images/NN_Desc3.png", caption="โค้ดสำหรับการโหลดข้อมูล")
    
    # Data Management
    with st.container():
        st.header("🔄 จัดการข้อมูล")
        
        st.markdown("""
        เราสร้างฟังก์ชัน prepare_data() ขึ้นมาเพื่อเตรียมข้อมูล โดยมันจะทำงานดังนี้
        1. เช็คว่าโฟลเดอร์ข้อมูลมีอยู่จริงมั้ย
        2. นับจำนวนภาพในแต่ละคลาส แล้วแสดงผลออกมา
        3. ใช้ Data Augmentation เพื่อเพิ่มความหลากหลายของข้อมูล โมเดลจะได้เรียนรู้หลายๆแบบ เช่น
           - Rescale → ปรับค่าพิกเซลให้อยู่ในช่วง 0-1
           - Rotation → หมุนภาพแบบสุ่ม (ไม่เกิน 20 องศา)
           - Shift / Shear / Zoom → ปรับรูปร่างภาพ
           - Horizontal flip → พลิกภาพในแนวนอน
        
        สุดท้าย เราสร้าง generator สำหรับชุด train และ validation โดยแบ่งเป็น 80:20
        """)
        
        st.image("./images/NN_Desc4.png", caption="การจัดการและเตรียมข้อมูล")
        st.image("./images/NN_Desc15.png", caption="แสดงผลการจัดการและเตรียมข้อมูล (ผมเรียกใช้ก่อน #main execution แต่เอามาไว้ตรงนี้ก่อน เพื่อให้เห็นภาพ)")
    
    # Create Model
    with st.container():
        st.header("🧠 สร้างโมเดล")
        
        st.markdown("""
        เราจะใช้ CNN + Transfer Learning โดยโหลด MobileNetV2 มาเป็นโมเดลพื้นฐาน
        - Freeze เลเยอร์ของโมเดลเดิมไว้ก่อน เพื่อไม่ให้ค่าถูกอัปเดต
        - เพิ่มเลเยอร์ใหม่เข้าไป เช่น
          - GlobalAveragePooling2D → ลดมิติของข้อมูล
          - Dense(128) + ReLU → Fully Connected Layer
          - Dropout(0.5) → ป้องกัน overfitting
          - Dense(64) + Dropout(0.3) → Fully Connected อีกชั้น
          - Dense(num_classes) → เลเยอร์สุดท้าย สำหรับจำแนก 4 คลาส

        จากนั้น เรา compile โมเดล โดยใช้
        - Adam Optimizer → ช่วยปรับค่าพารามิเตอร์ (จริงๆ มีหลายตัวให้เลือกใช้นะ เช่น Adagrad, AdamW, SGD)
        - Categorical Crossentropy Loss → เหมาะกับปัญหาจำแนกหลายคลาส
        - Accuracy Metric → วัดความแม่นยำ
        """)
        
        st.image("./images/NN_Desc5.png", caption="การสร้างโมเดลด้วย MobileNetV2")
    
    # Train Model
    with st.container():
        st.header("🏋️‍♂️ ฝึกสอนโมเดล")
        
        st.markdown("""
        ฟังก์ชัน train_model() จะทำหน้าที่เทรนโมเดล พร้อมใส่เทคนิคเสริม เช่น
        - Early Stopping → หยุดเทรนถ้าโมเดลไม่พัฒนาหลังจาก 5 epochs
        - ReduceLROnPlateau → ลดค่า learning rate ถ้า validation loss ไม่ดีขึ้น
        - ModelCheckpoint → บันทึกโมเดลที่ดีที่สุด
        """)
        
        st.image("./images/NN_Desc6.png", caption="การฝึกสอนโมเดล")
    
    # Fine-tuning
    with st.container():
        st.header("🔧 ปรับแต่งโมเดล (Fine-tuning)")
        
        st.markdown("""
        เราจะ ปลดล็อก 20 เลเยอร์สุดท้ายของ MobileNetV2 แล้วเทรนต่อ
        1. ลดค่า learning rate เป็น 1e-5
        2. เทรนเพิ่มอีก 10 epochs
        """)
        
        st.image("./images/NN_Desc7.png", caption="การ Fine-tuning โมเดล")
    
    # Evaluation
    with st.container():
        st.header("📊 ประเมินผล")
        
        st.markdown("""
        หลังจากเทรนเสร็จ เรามาประเมินผลกัน ไหนดูซิ อาการมันเป็นยังไง
        1. ใช้โมเดลทำนายคลาสของภาพในชุด validation
        2. ดู Confusion Matrix → เช็คว่าโมเดลทำนายพลาดจุดไหน
        3. ดู Classification Report → ดูค่า Precision, Recall และ F1-score
        """)
        
        st.image("./images/NN_Desc8.png", caption="การประเมินผลโมเดล")
    
    # Display Results
    with st.container():
        st.header("📈 แสดงผล")
        
        st.markdown("""
        เราแสดงผลการเทรนผ่านกราฟ
        1. plot_training_history(history) → ดูกราฟความแม่นยำ (Accuracy) และค่าความสูญเสีย (Loss)
        2. plot_training_history(fine_tune_history) → ดูกราฟความแม่นยำ (Accuracy) และค่าความสูญเสีย (Loss)
        3. plot_confusion_matrix() → ใช้ heatmap ดูว่าโมเดลทำนายพลาดตรงไหน
        """)
        
        st.image("./images/NN_Desc14.png", caption="main execution")
        st.image("./images/NN_Desc9.png", caption="ฟังก์ชันแสดงผลการเทรนโมเดล")
        st.image("./images/NN_Desc10.png", caption="ฟังก์ชันแสดงผลการเทรนโมเดลแบบ fine tune")
        st.image("./images/NN_Desc11.png", caption="การแสดงผลการเทรนโมเดล")
        st.image("./images/NN_Desc12.png", caption="การแสดงผลการเทรนโมเดลcบบ fine tune")
        st.image("./images/NN_Desc13.png", caption="การแสดงผลการเทรนโมเดลในรูปแบบ confusion matrix")
    
    # Test with Single Image
    with st.container():
        st.header("🖼️ ทดสอบกับภาพเดี่ยว")
        
        st.markdown("""
        เราสร้างฟังก์ชัน test_single_image() เพื่อทดสอบโมเดลกับภาพที่เราอัปโหลด
        1. โหลดภาพและปรับขนาด
        2. ทำนายคลาสและแสดงผล
        3. แสดงกราฟแท่งความน่าจะเป็นของแต่ละคลาส
        """)
        
        st.image("./images/NN_Desc16.png", caption="การทดสอบกับภาพเดี่ยว")
    
    # Summary
    with st.container():
        st.header("✅ สรุป")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            เอาล่ะหลังจากที่อ่ามาแบบยืดยาว มาสรุปสิ่งที่ทำแบบสั้นๆดีกว่า :)
            
            ✅ เตรียมข้อมูล  
            ✅ สร้างโมเดล  
            ✅ ฝึกสอนโมเดล  
            ✅ แสดงผลการเทรน  
            ✅ ปรับแต่งโมเดล (Fine-tune)  
            ✅ ประเมินผล  
            ✅ แสดง Confusion Matrix  
            ✅ บันทึกผลลัพธ์และโมเดลลง Google Drive  
            
            **สิ่งอย่างทำในอนาคต**  
            - ทำนายสภาพอากาศพร้อมกับบอก mood ของรูปนั้นด้วย
            """)
        
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p>🌦️ Predicting weather | Created by Nattawut Chongcasee</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()