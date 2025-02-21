Sonar Rock vs Mine Prediction - Machine Learning Project
=======================================================

📌 **Deskripsi Proyek:**
Proyek ini bertujuan untuk membangun model **Machine Learning** yang dapat 
membedakan antara **batu** (rock) dan **ranjau** (mine) berdasarkan data sonar.
Model yang digunakan adalah **Logistic Regression**, salah satu algoritma klasifikasi sederhana 
yang cocok untuk data biner.

🔹 **Dataset:**
Dataset yang digunakan adalah **Sonar Data**, yang berisi nilai pantulan sonar dari objek.
- **60 fitur pertama** adalah nilai numerik yang mewakili intensitas pantulan sonar.
- **Kolom ke-61 (label)** adalah kategori objek: "R" (Rock) atau "M" (Mine).


-----------------------------------------------------------
📂 **Penjelasan Code:**
-----------------------------------------------------------

1️⃣ **Import Library yang Diperlukan**
---------------------------------------
Kode ini mengimpor pustaka yang diperlukan, seperti pandas untuk manipulasi data,
sklearn untuk model machine learning, dan numpy untuk manipulasi array.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

📌 **Analogi:** Ini seperti menyiapkan alat-alat sebelum mulai mengerjakan proyek.


2️⃣ **Membaca Dataset**
-----------------------
Dataset dibaca menggunakan pandas dan disimpan dalam variabel `sonar_data`.

```python
sonar_data = pd.read_csv('/content/Copy of sonar data.csv', header=None)
```

📌 **Analogi:** Ini seperti membuka buku data sebelum kita mulai menganalisis.


3️⃣ **Eksplorasi Data**
-----------------------
- Melihat **5 baris pertama** dari dataset.
- Mengecek ukuran dataset.
- Melihat ringkasan statistik dataset.
- Melihat jumlah data per kategori "R" dan "M".

```python
sonar_data.head()
sonar_data.shape
sonar_data.describe()
sonar_data[60].value_counts()
```

📌 **Analogi:** Ini seperti melihat daftar isi buku dan membaca ringkasan isi sebelum mempelajarinya lebih dalam.


4️⃣ **Memisahkan Data dan Label**
----------------------------------
Data (X) dan label (Y) dipisahkan.

```python
x = sonar_data.drop(columns=60, axis=1)  # Fitur
y = sonar_data[60]  # Label
```

📌 **Analogi:** Ini seperti memisahkan soal ujian (X) dengan kunci jawabannya (Y).


5️⃣ **Membagi Dataset (Training & Testing)**
---------------------------------------------
Data dibagi menjadi **90% untuk training** dan **10% untuk testing**.

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)
```
- `test_size=0.1` → 10% data untuk pengujian.
- `stratify=y` → Membagi data secara proporsional berdasarkan jumlah "R" dan "M".
- `random_state=1` → Agar hasilnya konsisten setiap kali dijalankan.

📌 **Analogi:** Ini seperti membagi siswa menjadi kelompok latihan dan kelompok ujian.


6️⃣ **Melatih Model Logistic Regression**
-----------------------------------------
Model **Logistic Regression** digunakan untuk klasifikasi.

```python
model = LogisticRegression()
model.fit(x_train, y_train)
```

📌 **Analogi:** Ini seperti seorang siswa belajar dari buku dan latihan soal.


7️⃣ **Evaluasi Model**
----------------------
Mengukur akurasi model pada **training set** dan **testing set**.

```python
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on training data : ', training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy on Test data : ', test_data_accuracy)
```

📌 **Analogi:** Ini seperti memeriksa nilai ujian siswa setelah belajar.


8️⃣ **Memprediksi Data Baru**
-----------------------------
Menggunakan model yang telah dilatih untuk **memprediksi objek baru**.

```python
input_data = (...60 angka fitur...)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
```

📌 **Analogi:** Ini seperti seorang siswa mencoba menjawab soal baru setelah belajar.

Jika hasil prediksi adalah **"R"**, maka objek adalah batu. Jika **"M"**, maka objek adalah ranjau.
```python
if(prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')
```

📌 **Analogi:** Ini seperti guru memberi pertanyaan baru ke siswa, dan siswa menjawabnya.

-----------------------------------------------------------
📌 **Kesimpulan**
-----------------------------------------------------------
✅ Model **Logistic Regression** digunakan untuk klasifikasi objek sonar.

✅ Dataset mengandung **60 fitur numerik** dan **1 kolom target** ('R' atau 'M'). 

✅ Model dilatih menggunakan **90% data**, diuji dengan **10% data**.

✅ Akurasi model dihitung untuk menilai performanya.

✅ Model bisa digunakan untuk **memprediksi objek baru**.

📌 **Analogi keseluruhan:**
- Dataset = Buku latihan soal.
- Model = Siswa yang belajar.
- Training = Siswa latihan soal.
- Testing = Siswa ujian.
- Prediksi = Siswa menjawab soal baru.
- Akurasi = Nilai ujian siswa.