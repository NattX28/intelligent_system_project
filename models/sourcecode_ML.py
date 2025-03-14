import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# โหลดข้อมูล
file_path = "/content/drive/MyDrive/SpotifyFeatures.csv"
df = pd.read_csv(file_path)

# ดู head ของข้อมูล
df.head()

# ดู tail ของข้อมูล
df.tail()

# ดูว่ามีค่า null ไหม ถ้ามี มีกี่อัน
df.isnull().sum()

plt.figure(figsize=(8,15))
sns.boxplot(data=df[['acousticness', 'danceability', 'energy', 'instrumentalness',
                     'key', 'liveness', 'loudness', 'mode', 'speechiness',
                     'tempo', 'time_signature', 'valence']], orient='h')
plt.title('Distribution of Spotify Audio Features')
plt.tight_layout()
plt.show()

# ตรวจสอบและลบ Missing Values
df.dropna(inplace=True)

# เลือกเฉพาะคอลัมน์ที่ใช้เป็น Features และ Target
features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
            'key', 'liveness', 'loudness', 'mode', 'speechiness',
            'tempo', 'time_signature', 'valence']
target = 'popularity'

# แปลงข้อมูลที่เป็นหมวดหมู่ (เช่น key, mode, time_signature)
df['key'] = LabelEncoder().fit_transform(df['key'])
df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0})
df['time_signature'] = LabelEncoder().fit_transform(df['time_signature'])

# จัดการ Outliers โดยใช้ IQR
Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[~((df[features] < lower_bound) | (df[features] > upper_bound)).any(axis=1)]

# แยกข้อมูล Train และ Test
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Standardization ด้วย StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# เทรน Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

# เทรน Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=15, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# ประเมินผลลัพธ์
print("Linear Regression Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lin):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lin)):.2f}")

print("\nRandom Forest Regressor Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")

# สร้างกราฟเปรียบเทียบผลลัพธ์
plt.figure(figsize=(10, 5))
sns.histplot(y_test, color='blue', label='Actual', kde=True, alpha=0.5)
sns.histplot(y_pred_rf, color='red', label='Predicted', kde=True, alpha=0.5)
plt.xlabel("Popularity Score")
plt.ylabel("Frequency")
plt.title("Actual vs Predicted Popularity Distribution")
plt.legend()
plt.show()

# บันทึกโมเดลและ Scaler
with open("spotify_popularity_lr.pkl", "wb") as f:
    pickle.dump(lin_reg, f)

with open("spotify_popularity_rf.pkl", "wb") as f:
    pickle.dump(rf_reg, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("โมเดลถูกบันทึกเรียบร้อยแล้ว!")