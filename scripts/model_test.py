import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Veriyi yükle
data = pd.read_json('C:/Users/Ensar/Desktop/OneriSistemi/data/student_data.json')

# Kategorik verileri sayısal verilere dönüştür
data['part_time_job'] = data['part_time_job'].map({True: 1, False: 0})
data['extracurricular_activities'] = data['extracurricular_activities'].map({True: 1, False: 0})

# Eksik verileri işleme
data.replace('Unknown', np.nan, inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Yeni bir hedef değişken oluştur: Notların ortalaması
data['average_score'] = data[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].mean(axis=1)

# Özellik ve hedef değişkenleri ayır
X = data[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score', 'weekly_self_study_hours', 'part_time_job', 'absence_days', 'extracurricular_activities']]
y = data['average_score']

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# Özelliklerin etkilerini inceleyin
coefficients = model.coef_
feature_names = X.columns

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef}")
