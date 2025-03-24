import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Veri yükleme
data_path = 'Student Depression Dataset.csv'
df = pd.read_csv(data_path)

# Veri keşfi
print("Veri setinin ilk 5 satırı:")
print(df.head())
print("\nSütun bilgileri:")
print(df.info())
print("\nSınıf dağılımı:")
print(df['Depression'].value_counts())

# Eksik veri kontrolü
print("\nEksik veri sayısı:")
print(df.isnull().sum())

# Eksik veri doldurma (Sürekli -> Ortalama, Kategorik -> Mod)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'O':  # Kategorik veri
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:  # Sürekli veri
            df[col].fillna(df[col].mean(), inplace=True)

# Kategorik değişkenleri sayısala dönüştürme
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'O':  # Eğer sütun kategorik ise
        df[col] = le.fit_transform(df[col])

# Özellikleri ve hedef değişkeni ayırma
X = df.drop(columns=['Depression'])
y = df['Depression']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Özellik ölçekleme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Scikit-learn Logistic Regression modeli
time_start = time.time()
sklearn_model = LogisticRegression()
sklearn_model.fit(X_train, y_train)
sklearn_train_time = time.time() - time_start

# Tahmin ve değerlendirme
time_start = time.time()
sklearn_predictions = sklearn_model.predict(X_test)
sklearn_test_time = time.time() - time_start

# Karmaşıklık matrisi ve metrikler
cm = confusion_matrix(y_test, sklearn_predictions)
print("\nKarmaşıklık Matrisi:")
print(cm)

# Confusion Matrix görselleştirme
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Eksenleri etiketleme
classes = ['Not Depressed', 'Depressed']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Sayıları ekleme
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen Değer')
plt.tight_layout()
plt.show()

# Doğruluk Skoru ve sınıflandırma raporu
print("\nDoğruluk Skoru:", accuracy_score(y_test, sklearn_predictions))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, sklearn_predictions))

# Model zamanlarını yazdırma
print(f"Scikit-learn Modeli Eğitim Süresi: {sklearn_train_time:.4f} saniye")
print(f"Scikit-learn Modeli Test Süresi: {sklearn_test_time:.4f} saniye")
