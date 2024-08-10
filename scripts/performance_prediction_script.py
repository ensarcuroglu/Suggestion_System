import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Veriyi yükle
data = pd.read_json('C:/Users/Ensar/Desktop/OneriSistemi/data/student_data.json')

# Kategorik verileri sayısala çevirme
data['part_time_job'] = data['part_time_job'].map({True: 1, False: 0})
data['extracurricular_activities'] = data['extracurricular_activities'].map({True: 1, False: 0})

# Eksik verileri işleme
data.replace('Unknown', np.nan, inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Yeni bir hedef değişken oluştur: Notların ortalaması
data['average_score'] = data[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].mean(axis=1)

# Yeni hedef değişkeni oluşturma
# Ağırlıklı performans puanı: Not ortalamasının yarısı + diğer değişkenlerin etkisi
data['performance_score'] = (
    data['average_score'] * 0.5
    - data['part_time_job'] * 0.2
    - data['absence_days'] * 0.2
    + data['extracurricular_activities'] * 0.1
    + data['weekly_self_study_hours'] * 0.3
)

# Özellik ve hedef değişkenleri ayırma
X = data[['average_score', 'part_time_job', 'absence_days', 'extracurricular_activities', 'weekly_self_study_hours']]
y = data['performance_score']


# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor modelini oluşturma
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Modeli eğitme
model.fit(X_train, y_train)



# Performans kategorisi belirleyen fonksiyon
def classify_performance(score, q1, q3):
    if score >= q3:
        return "Performansı artışta"
    elif score >= q1:
        return "Performansı stabil"
    else:
        return "Performansı düşüşte"

# Performans kategorisini belirlemek için gerekli olan çeyrek değerler
q1 = data['performance_score'].quantile(0.25)
q3 = data['performance_score'].quantile(0.75)


# Model ve fonksiyonu bir sınıf ile birleştirme
class PerformanceModel:
    def __init__(self, model, q1, q3):
        self.model = model
        self.q1 = q1
        self.q3 = q3

    def predict(self, X):
        return self.model.predict(X)

    def classify(self, score):
        return classify_performance(score, self.q1, self.q3)


# Sınıfın bir örneğini oluşturma
performance_model = PerformanceModel(model, q1, q3)

# Modeli kaydetme
joblib.dump(performance_model, 'C:/Users/Ensar/Desktop/OneriSistemi/models/student_performance_model.pkl')
