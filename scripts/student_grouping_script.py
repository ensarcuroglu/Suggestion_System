import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Veriyi yükle
data = pd.read_json('C:/Users/Ensar/Desktop/OneriSistemi/data/student_data.json')

# Kategorik verileri sayısala çevirme
data['part_time_job'] = data['part_time_job'].map({True: 1, False: 0})
data['extracurricular_activities'] = data['extracurricular_activities'].map({True: 1, False: 0})

# Eksik verileri işleme
data.replace('Unknown', np.nan, inplace=True)

# Sayısal olmayan değerleri sayısal değerlere dönüştür
data = data.apply(pd.to_numeric, errors='coerce')

# Eksik verileri ortalamalarla doldur
data.fillna(data.mean(numeric_only=True), inplace=True)

# Yeni bir hedef değişken oluştur: Notların ortalaması
data['average_score'] = data[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].mean(axis=1)

# Özellik seçimleri
X = data[['average_score', 'part_time_job', 'absence_days', 'extracurricular_activities', 'weekly_self_study_hours']]

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method ile uygun k değerini belirleme
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Kümeleri belirleme
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)
data['cluster'] = kmeans.labels_

# Grafik fonksiyonları
def plot_elbow_method():
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method For Optimal k')
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.show()

def plot_scatter():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='average_score', y='absence_days', hue='cluster', palette='Set1', s=100, alpha=0.7)
    plt.title('Kümelere Göre Öğrenci Dağılımı')
    plt.xlabel('Ortalama Not')
    plt.ylabel('Devamsızlık Günleri')
    plt.legend(title='Küme')
    plt.show()

def plot_box_plots():
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, x='cluster', y='average_score', hue='cluster', palette='Set1', legend=False)
    plt.title('Kümelere Göre Ortalama Not Dağılımı')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=data, x='cluster', y='absence_days', hue='cluster', palette='Set1', legend=False)
    plt.title('Kümelere Göre Devamsızlık Günleri Dağılımı')

    plt.show()

def plot_pairplot():
    sns.pairplot(data, vars=['average_score', 'absence_days', 'weekly_self_study_hours'], hue='cluster', palette='Set1')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap():
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[['average_score', 'absence_days', 'weekly_self_study_hours', 'part_time_job', 'extracurricular_activities']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Özellikler Arası Korelasyon Matrisi')
    plt.show()

def plot_cluster_centers():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='average_score', y='absence_days', hue='cluster', palette='Set1', s=100, alpha=0.7)
    centers = kmeans.cluster_centers_
    scaled_centers = scaler.inverse_transform(centers)  # Orijinal ölçeklere dönüştür
    plt.scatter(scaled_centers[:, 0], scaled_centers[:, 1], s=300, c='red', marker='X', label='Küme Merkezleri')
    plt.title('Kümelere Göre Öğrenci Dağılımı ve Küme Merkezleri')
    plt.xlabel('Ortalama Not')
    plt.ylabel('Devamsızlık Günleri')
    plt.legend(title='Küme')
    plt.show()

# Kullanıcıdan hangi grafiği görmek istediğini seçmesini isteyin
print("Gösterilecek grafiği seçin:")
print("1: Elbow Method grafiği")
print("2: Öğrenci Dağılımı Scatter Plot")
print("3: Ortalama Not ve Devamsızlık Günleri Box Plot")
print("4: Özellikler Arası Korelasyon Matrisi Heatmap")
print("5: Küme Merkezleri Scatter Plot")
choice = input("Seçiminizi yapın (1-5): ")

# Seçime göre grafiği göster
if choice == '1':
    plot_elbow_method()
elif choice == '2':
    plot_scatter()
elif choice == '3':
    plot_box_plots()
elif choice == '4':
    plot_correlation_heatmap()
elif choice == '5':
    plot_cluster_centers()
else:
    print("Geçersiz seçim. Lütfen 1-5 arasında bir değer girin.")

# Modeli ve scaler'ı kaydet
joblib.dump(kmeans, 'C:/Users/Ensar/Desktop/OneriSistemi/models/kmeans_model.pkl')
joblib.dump(scaler, 'C:/Users/Ensar/Desktop/OneriSistemi/models/scaler.pkl')