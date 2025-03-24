import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]] += 1
    return matrix

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

# Eksik veri kontrolü ve doldurma
print("\nEksik veri sayısı:")
print(df.isnull().sum())

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'O':  # Kategorik veri
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:  # Sürekli veri
            df[col].fillna(df[col].mean(), inplace=True)

# Kategorik değişkenleri sayısala dönüştürme
for col in df.columns:
    if df[col].dtype == 'O':
        unique_values = df[col].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_map)

# Özellikleri ve hedef değişkeni ayırma
X = df.drop(columns=['Depression']).values
y = df['Depression'].values

# Veriyi normalize etme
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Veriyi eğitim ve test setlerine ayırma
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Model eğitimi
print("\nModel eğitiliyor...")
time_start = time.time()
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)
train_time = time.time() - time_start

# Tahmin
time_start = time.time()
predictions = model.predict(X_test)
test_time = time.time() - time_start

# Sonuçları değerlendirme
acc = accuracy(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f"\nDoğruluk: {acc:.4f}")
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

# Model zamanlarını yazdırma
print(f"\nModel Eğitim Süresi: {train_time:.4f} saniye")
print(f"Model Test Süresi: {test_time:.4f} saniye")
