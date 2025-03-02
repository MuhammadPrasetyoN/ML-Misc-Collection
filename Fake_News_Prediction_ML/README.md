# 📌 Fake News Detection dengan Logistic Regression

## 📰 Deskripsi Proyek
Proyek ini bertujuan untuk membangun model machine learning yang dapat mendeteksi berita palsu (fake news) berdasarkan teks berita. Dataset yang digunakan terdiri dari berita dengan label `0` (berita asli) dan `1` (berita palsu). 

Algoritma yang digunakan untuk klasifikasi adalah **Logistic Regression**. Sebelum dilakukan pelatihan model, teks berita diolah menggunakan teknik **stemming** dan dikonversi ke bentuk numerik menggunakan **TF-IDF Vectorization**.

## 🛠️ Dependensi
Sebelum menjalankan proyek ini, pastikan Anda telah menginstal pustaka berikut:
```python
import pandas as pd
import numpy as np
import re  # Regular Expression
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk
```

---
## 1. 🧹 Data Preprocessing

### 🔍 a.  Mengecek dan Membersihkan Data
- Dataset dimuat menggunakan `pandas.read_csv()`.
- Data yang hilang (`NaN`) diisi dengan string kosong.
- Judul berita dan isi berita digabungkan menjadi satu kolom baru (`header_content`).

```python
news_dataset = pd.read_csv("train.csv")
news_dataset = news_dataset.fillna('')
news_dataset['header_content'] = news_dataset['author'] + ' ' + news_dataset['title']
```

### ✂️ b. Stemming
Stemming adalah proses mengubah kata menjadi bentuk dasarnya. Dalam proyek ini, stemming dilakukan menggunakan **PorterStemmer** dari pustaka `nltk`. Stemming bertujuan untuk mengurangi variasi kata yang memiliki makna sama, misalnya "running" dan "run" dianggap sebagai satu kata dasar.

#### Implementasi Stemming:
```python
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

port_stem = PorterStemmer()

def stemmer(content):
    # Menghapus karakter non-huruf
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    
    # Mengubah teks menjadi huruf kecil
    stemmed_content = stemmed_content.lower()
    
    # Memisahkan kata-kata dalam teks
    stemmed_content = stemmed_content.split()
    
    # Melakukan stemming dan menghapus stopwords (kata umum yang tidak bermakna spesifik)
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    
    # Menggabungkan kata yang telah diproses menjadi teks kembali
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content
```

### 🔢 c. Vectorization dengan TF-IDF
Karena model machine learning tidak dapat bekerja langsung dengan data teks, teks perlu diubah menjadi format numerik. Teknik yang digunakan adalah **TF-IDF (Term Frequency - Inverse Document Frequency)**, yang memberikan bobot lebih tinggi pada kata-kata yang unik di setiap dokumen dan bobot lebih rendah pada kata-kata yang sering muncul di banyak dokumen.

#### Implementasi Vectorization:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer.fit(X)  # X adalah teks yang telah melalui proses stemming
X = vectorizer.transform(X)  # Mengubah teks menjadi representasi numerik
```

Setelah proses ini, setiap berita akan direpresentasikan dalam bentuk vektor dengan dimensi sesuai jumlah kata unik dalam dataset.

**Contoh:**
_Input:_ `"Fake news spread quickly"`

_Output (TF-IDF Vector):_
```
  (0, 267)   0.2701
  (0, 2483)  0.3676
  (0, 2959)  0.2468
```

---
## 📊 2. Pemisahan Data
Dataset dibagi menjadi **training set** dan **testing set** agar model dapat diuji performanya sebelum digunakan pada data baru.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

---
## 🤖 3. Pelatihan Model dengan Logistic Regression
Model Logistic Regression digunakan karena cocok untuk tugas klasifikasi biner seperti deteksi fake news.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)  # Melatih model
```

---
## 📌  Kesimpulan
Dalam proyek ini, berita dikonversi menjadi bentuk numerik menggunakan **stemming** dan **TF-IDF vectorization** sebelum dilatih menggunakan **Logistic Regression**. Model ini mampu memprediksi apakah suatu berita termasuk berita palsu atau asli berdasarkan teksnya.