# 🏥 Machine Learning untuk Deteksi Diabetes

## 📌 Deskripsi Proyek
Proyek ini menggunakan **Support Vector Machine (SVM)** untuk mendeteksi diabetes berdasarkan dataset kesehatan.
Dataset ini memiliki **8 fitur** sebagai input dan **1 label output**, yaitu:
- `1` (Diabetes)
- `0` (Tidak Diabetes)

## 🔧 Teknologi yang Digunakan
- 🐍 **Python**
- 📊 **Pandas** untuk manipulasi dataset
- 🔢 **NumPy** untuk komputasi numerik
- 🤖 **Scikit-learn** untuk preprocessing, pelatihan model, dan evaluasi

---
## 📂 Struktur Kode

### 1️⃣ Import Library 📦  
Mengimpor library yang diperlukan:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

### 2️⃣ Membaca Dataset 📑  
Dataset dibaca menggunakan Pandas:
```python
df_diabet = pd.read_csv('diabetes.csv')
print(df_diabet.head())  # Menampilkan 5 baris pertama
```

### 3️⃣ Eksplorasi Data 🔍  
- **Cek jumlah data & fitur**: `df_diabet.shape`
- **Distribusi label**: `df_diabet['Outcome'].value_counts()`
- **Rata-rata fitur berdasarkan label**:
```python
df_diabet.groupby('Outcome').mean()
```

### 4️⃣ Preprocessing Data 🛠  
- **Pisahkan fitur (X) dan label (Y)**:
```python
df_diabet_x = df_diabet.drop(columns='Outcome', axis=1)
df_diabet_y = df_diabet['Outcome']
```
- **Standarisasi fitur dengan `StandardScaler`**:
```python
scaler = StandardScaler()
df_diabet_x_standarized = scaler.fit_transform(df_diabet_x)
```

### 5️⃣ Membagi Dataset 📊  
Data dibagi menjadi **80% data latih** dan **20% data uji**:
```python
X_train, X_test, Y_train, Y_test = train_test_split(
    df_diabet_x_standarized, df_diabet_y, test_size=0.2, stratify=Y, random_state=2
)
```
#### 📌 Penjelasan `stratify` dan `random_state`
- **`stratify=Y`**: Menjaga proporsi label diabetes dan tidak diabetes tetap seimbang dalam data latih dan uji.
- **`random_state=2`**: Memastikan hasil pembagian data **tetap sama** setiap kali kode dijalankan.

### 6️⃣ Melatih Model SVM 🤖  
Model **SVM dengan kernel linear** digunakan:
```python
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, Y_train)
```

### 7️⃣ Evaluasi Model 📈  
- **Akurasi pada data latih**:
```python
X_train_pred = svm_classifier.predict(X_train)
training_accuracy = accuracy_score(X_train_pred, Y_train)
print('Akurasi pada data latih:', training_accuracy)
```
- **Akurasi pada data uji**:
```python
X_test_pred = svm_classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_pred, Y_test)
print('Akurasi pada data uji:', test_accuracy)
```

### 8️⃣ Prediksi dengan Data Baru 🔮  
Gunakan model untuk memprediksi satu data baru:
```python
input_data = np.array([11,143,94,33,146,36.6,0.254,51])
input_data_scaled = scaler.transform(input_data.reshape(1, -1))
prediction = svm_classifier.predict(input_data_scaled)
print('Prediksi:', 'Diabetes' if prediction[0] == 1 else 'Tidak Diabetes')
```

### 9️⃣ Prediksi dengan Beberapa Data Sekaligus 🔢  
Gunakan model untuk memprediksi beberapa data sekaligus:
```python
data_input = [
    [10,122,78,31,0,27.6,0.512,45],
    [4,103,60,33,192,24,0.966,33],
    [11,138,76,0,0,33.2,0.42,35],
    [11,143,94,33,146,36.6,0.254,51],
    [8,176,90,34,300,33.7,0.467,58]
]
data_input_scaled = scaler.transform(np.asarray(data_input))
predictions = svm_classifier.predict(data_input_scaled)
for i, pred in enumerate(predictions):
    print(f"Data ke-{i+1}: {'Diabetes' if pred == 1 else 'Tidak Diabetes'}")
```

---
## 🎯 Hasil yang Diharapkan
Model akan mengklasifikasikan apakah seseorang menderita **diabetes** atau **tidak** berdasarkan data kesehatan mereka.
- **Akurasi data latih** menunjukkan seberapa baik model belajar.
- **Akurasi data uji** mengukur kemampuan model terhadap data yang belum pernah dilihat sebelumnya.

## ⚠️ Catatan Penting
- Pastikan dataset `diabetes.csv` tersedia sebelum menjalankan kode.
- **Gunakan `StandardScaler` yang sama** untuk standarisasi data baru sebelum prediksi.
- SVM dengan **kernel linear** dipilih karena cocok untuk data dengan pemisahan linier.
