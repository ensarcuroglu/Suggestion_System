import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from joblib import dump, load

# 1) Veri Yükleme
# -----------------------------------------------------------------
data = pd.read_json('C:/Users/Ensar/Desktop/OneriSistemi/data/student_data.json')
df = pd.DataFrame(data)

# 2) Hedef Değişken (Target) Tanımlama
# -----------------------------------------------------------------
score_cols = [
    "math_score",
    "history_score",
    "physics_score",
    "chemistry_score",
    "biology_score",
    "english_score",
    "geography_score",
]
df["performance_score"] = df[score_cols].mean(axis=1)

# 3) Modelde Kullanılacak Özellikleri (Features) ve Hedefi (Target) Seçme
# -----------------------------------------------------------------
X = df[
    [
        "gender",
        "part_time_job",
        "absence_days",
        "extracurricular_activities",
        "weekly_self_study_hours",
        "career_aspiration",
    ]
]
y = df["performance_score"]

# 4) Kategorik ve Sayısal Değişkenleri Belirleme
# -----------------------------------------------------------------
categorical_features = ["gender", "career_aspiration"]
numeric_features = ["part_time_job", "absence_days", "extracurricular_activities", "weekly_self_study_hours"]

# 5) Ön İşleme Adımları
# -----------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough"
)

# 6) Pipeline Oluşturma ve Model Seçimi
# -----------------------------------------------------------------
model_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# 7) Veri Setini Eğitim/Test Olarak Ayırma
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 8) Modeli Eğitme
# -----------------------------------------------------------------
model_pipeline.fit(X_train, y_train)

# Modeli kaydet
model_path = "../models/student_performance_model.joblib"
dump(model_pipeline, model_path)
print(f"Model '{model_path}' dosyasına başarıyla kaydedildi.")

# 9) Model Performansını Değerlendirme
# -----------------------------------------------------------------
y_pred = model_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Test MSE :", mse)
print("Test RMSE:", rmse)
print("Test R^2 :", r2)

# 10) Yeni Veriler Üzerinde Tahmin
# -----------------------------------------------------------------
new_student = pd.DataFrame({
    "gender": ["F"],
    "part_time_job": [0],
    "absence_days": [2],
    "extracurricular_activities": [1],
    "weekly_self_study_hours": [5],
    "career_aspiration": ["doctor"],
})

# Yeni öğrenci için tahmin
predicted_performance = model_pipeline.predict(new_student)
print("\nYeni Öğrencinin Tahmini Performansı:", predicted_performance[0])
