import joblib
import pandas as pd

loaded_model = joblib.load('C:/Users/Ensar/Desktop/OneriSistemi/models/student_risk_model.pkl')

# Test etmek için bir öğrenci verisi girişi oluşturma
new_student = pd.DataFrame({
    'absence_days': [5],
    'weekly_self_study_hours': [24],
    'part_time_job': [1],  # False = 0, True = 1
    'extracurricular_activities': [1],  # False = 0, True = 1
    'math_score': [85],
    'biology_score': [90],
    'physics_score': [80],
    'chemistry_score': [88]
})

# Kategorik sütunları sayısal hale getir
new_student['part_time_job'] = new_student['part_time_job'].astype(int)
new_student['extracurricular_activities'] = new_student['extracurricular_activities'].astype(int)

# Tahmin yapma
risk_prediction = loaded_model.predict(new_student)


# Sonucu gösterme
print(f"Öğrenci risk seviyesi: {risk_prediction[0]}")