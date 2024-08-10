import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# PerformanceModel sınıfını tanımla
class PerformanceModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def classify(self, score):
        q1 = 40.557143  # Quartile 1 value
        q3 = 46.657143  # Quartile 3 value
        if score >= q3:
            return "Performansı artışta"
        elif score >= q1:
            return "Performansı stabil"
        else:
            return "Performansı düşüşte"



# Modeli yükle
performance_model = joblib.load('C:/Users/Ensar/Desktop/OneriSistemi/models/student_performance_model.pkl')

# Yeni veri
new_data = pd.DataFrame({
    'average_score': [80],
    'part_time_job': [1],
    'absence_days': [10],
    'extracurricular_activities': [1],
    'weekly_self_study_hours': [25]
})

# Tahmin yap
predicted_score = performance_model.predict(new_data)

# Performans sınıflandırması yap
performance_category = performance_model.classify(predicted_score[0])

print(f"Predicted Score: {predicted_score[0]}")
print(f"Performance Category: {performance_category}")