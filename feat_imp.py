import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Veri Yükleme
data = pd.read_csv("electricity.csv")

# Hedef değişken encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])  # UP -> 1, DOWN -> 0

# Tarih sütununu datetime formatına çevir ve epoch zaman damgasına dönüştür
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].apply(lambda x: x.timestamp())

# Kategorik veriyi encode et (ör. day sütunu)
data['day'] = le.fit_transform(data['day'])

# Girdi ve Çıktı Ayrımı
X = data[['date', 'day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']]
y = data['class']

# Veri Setini Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Modeli ile Feature Importance
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Özellik Önem Skorları
feature_importances = model.feature_importances_
feature_names = X.columns

# Skorları bir DataFrame'e dök
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Önem Skorlarını Görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.show()

# Önem skoru tablosunu yazdır
print("Feature Importances:")
print(importance_df)
