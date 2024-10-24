import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Veriyi yükle
data = pd.read_json('C:/Users/Ensar/Desktop/OneriSistemi/data/student_data.json')

# 1. Veriyi ön işleme
# Gerekli sütunları seçelim
features = ['absence_days', 'weekly_self_study_hours', 'part_time_job', 'extracurricular_activities', 'math_score', 'biology_score', 'physics_score', 'chemistry_score']
X = data[features]

# Yüksek riskli öğrencileri belirlemek için örnek bir hedef sütun oluşturalım
# Devamsızlık günleri 5'ten fazla ve haftalık çalışma saati 10'dan az ise yüksek risk
data['risk_level'] = data.apply(lambda row: 'high' if row['absence_days'] > 5 and row['weekly_self_study_hours'] < 10 else('medium' if row['absence_days'] > 3 else 'low'), axis=1)
y = data['risk_level']

# Kategorik verileri işleme (True/False olan sütunları 0/1 yapalım)
X.loc[:, 'part_time_job'] = X['part_time_job'].astype(int)
X.loc[:, 'extracurricular_activities'] = X['extracurricular_activities'].astype(int)

# 2. Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Modeli oluşturma ve eğitme
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Modeli test etme
y_pred = model.predict(X_test)

# Doğruluk skorunu ve diğer metrikleri yazdırma
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Modeli kaydetme
joblib.dump(model, 'C:/Users/Ensar/Desktop/OneriSistemi/models/student_risk_model.pkl')



