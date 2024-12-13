import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Performans Değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
