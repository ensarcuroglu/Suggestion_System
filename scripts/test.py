import pandas as pd
import numpy as np

# Veriyi yükle
data = pd.read_json('C:/Users/Ensar/Desktop/OneriSistemi/data/student_data.json')

data['part_time_job'] = data['part_time_job'].map({True: 1, False: 0})
data['extracurricular_activities'] = data['extracurricular_activities'].map({True: 1, False: 0})

# Eksik verileri işleme
data.replace('Unknown', np.nan, inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)  # Sayısal sütunların ortalamasıyla doldur

# Yeni bir hedef değişken oluştur: Notların ortalaması
data['average_score'] = data[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].mean(axis=1)

data['point'] = data['average_score']* 0.5 - data['part_time_job'] * 0.1 - data['absence_days'] * 0.1 - data['extracurricular_activities'] * 0.1 + data['weekly_self_study_hours'] * 0.2

print(data['point'])