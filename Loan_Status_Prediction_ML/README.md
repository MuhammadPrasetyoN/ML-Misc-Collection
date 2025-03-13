# 🏦 Prediksi Status Peminjaman (Loan Prediction) menggunakan SVM

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk memprediksi apakah seseorang layak mendapatkan pinjaman berdasarkan berbagai faktor seperti status pernikahan, tingkat pendidikan, area tempat tinggal, dan faktor lainnya. Model yang digunakan dalam proyek ini adalah **Support Vector Machine (SVM)**, yang dikenal efektif dalam klasifikasi data dengan margin optimal.

Dataset yang digunakan berasal dari data peminjaman bank yang memiliki fitur kategori dan numerik. Oleh karena itu, diperlukan **pra-pemrosesan data** untuk mengonversi data kategori menjadi format numerik sebelum diterapkan ke model.

---

## 🔧 Tahapan Pemrosesan Data

1. **Pembersihan Data**
   - Mengecek data yang hilang (missing values) dan menghapusnya jika diperlukan.
   
2. **Encoding Data Kategori**
   - Mengonversi data kategori menjadi angka agar dapat diproses oleh model Machine Learning.

3. **Membagi Dataset**
   - Memisahkan data menjadi **data latih (train set)** dan **data uji (test set)**.
   
4. **Melatih Model SVM**
   - Melatih model menggunakan algoritma **Support Vector Machine** dengan kernel linear.
   
5. **Evaluasi Model**
   - Mengukur performa model dengan **Akurasi**, **F1-Score**, dan **Classification Report**.

---

## 🔠 Encoding Data Kategori
Sebelum membangun model Machine Learning, kita harus mengubah fitur kategori menjadi numerik. Ada dua metode yang digunakan dalam proyek ini:

### 🎯 1. Label Encoding
- **Digunakan untuk fitur kategori yang bersifat biner** (hanya memiliki dua nilai unik).
- Mengubah nilai kategori menjadi angka **0 dan 1**.
- Contoh:

```python
loan_df.replace({
    'Married': {'Yes': 1, 'No': 0},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Education': {'Not Graduate': 0, 'Graduate': 1}
}, inplace=True)
```

📌 **Kenapa Label Encoding?**
- Karena fitur ini hanya memiliki dua kategori, cukup diubah ke **0** dan **1** tanpa menambah dimensi data.

### 🎯 2. One-Hot Encoding
- **Digunakan untuk fitur kategori dengan lebih dari dua nilai unik**.
- Mengubah setiap kategori menjadi kolom baru dengan nilai **0 atau 1**.
- Contoh:

```python
loan_df = pd.get_dummies(loan_df, columns=['Property_Area'])
```

📌 **Kenapa One-Hot Encoding?**
- Karena fitur seperti `Property_Area` memiliki lebih dari dua kategori (**Rural, Semiurban, Urban**), jika hanya menggunakan Label Encoding, model bisa menganggap ada hubungan numerik di antara kategori, padahal sebenarnya tidak.

---

## 🛠️ Model Machine Learning
Model yang digunakan dalam proyek ini adalah **Support Vector Machine (SVM) dengan kernel linear**. Model ini dipilih karena mampu melakukan klasifikasi dengan **margin optimal**, sehingga dapat bekerja dengan baik untuk dataset dengan dimensi tinggi.

```python
from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

Setelah model dilatih, dilakukan prediksi dan evaluasi menggunakan **Accuracy Score** dan **F1-Score**:

```python
from sklearn.metrics import accuracy_score, f1_score
Y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
print("Akurasi:", accuracy)
print("F1-Score:", f1)
```

---

## 📊 Hasil Visualisasi Data
Dataset divisualisasikan menggunakan **Seaborn** untuk memahami pola hubungan antara fitur dengan status peminjaman.

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Education', hue='Loan_Status', data=loan_df)
plt.show()
```

Visualisasi membantu dalam **analisis eksplorasi data (EDA)** untuk memahami bagaimana faktor tertentu mempengaruhi peluang seseorang mendapatkan pinjaman.

---

## 🎯 Kesimpulan
- **Encoding kategori sangat penting** agar model Machine Learning dapat bekerja dengan baik.
- **Label Encoding** digunakan untuk fitur kategori biner, sedangkan **One-Hot Encoding** digunakan untuk fitur dengan lebih dari dua kategori.
- Model **SVM** dengan kernel linear berhasil digunakan untuk memprediksi status peminjaman dengan tingkat akurasi yang cukup baik.
