import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Veri Yükleme
data = pd.read_csv("electricity.csv")

# Hedef değişken encoding
le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])  # UP -> 1, DOWN -> 0

# Tarih sütununu datetime formatına çevir ve epoch zaman damgasına dönüştür
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].apply(lambda x: x.timestamp())

# Kategorik veriyi encode et (ör. day sütunu)
data['day'] = le.fit_transform(data['day'])

# Girdi ve Çıktı Ayrımı
X = data[['date', 'day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']].values
y = data['class'].values

# Tüm verinin sayısal olduğunu kontrol et
X = X.astype('float32')

# Veri Setini Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Oluşturma
model = Sequential([
    Dense(16, activation='relu', input_dim=X.shape[1]),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # İkili sınıflandırma için sigmoid
])

# Model Derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Eğitimi
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Performans Değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)  # 0.5 eşik değeri kullanarak ikili sınıflandırma

# Metrikler
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
entropy = log_loss(y_test, y_pred)

# Sonuçları tablo halinde sunmak için
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'R²': r2,
    'MSE': mse,
    'Entropy': entropy
}

# Tabloyu oluşturma
metrics_df = pd.DataFrame(metrics, index=[0])

# Performans Raporu
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
print("\nModel Performans Metrikleri:")
print(metrics_df)

# Performans metrikleri verisi
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'R²', 'MSE', 'Entropy'],
    'Score': [accuracy, precision, recall, f1, recall, r2, mse, entropy]
}

# DataFrame oluşturma
df_metrics = pd.DataFrame(metrics)

# Grafik çizme
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Score', y='Metric', data=df_metrics, palette='viridis')

# Başlık ve etiketler
plt.title('Model Performans Metrikleri', fontsize=16)
plt.xlabel('Değer', fontsize=12)
plt.ylabel('Metrik', fontsize=12)

# Her barın üzerine metrik değerini yazdırma
for p in ax.patches:
    ax.annotate(f'{p.get_width():.4f}', 
                (p.get_x() + p.get_width() * 0.9, p.get_y() + p.get_height() / 2),
                ha='center', va='center', fontsize=12, color='black')

# Gösterme
plt.show()
