# 🏦 Deposit Prediction Using Machine Learning

## 📌 Project Overview
Proyek ini bertujuan untuk memprediksi apakah seorang nasabah akan melakukan deposit atau tidak berdasarkan dataset perbankan. Model yang digunakan adalah **Random Forest** dan **XGBoost**.

## 📂 Dataset
Dataset berisi berbagai informasi tentang pelanggan, termasuk usia, status pernikahan, pekerjaan, saldo rekening, dan lainnya.

## 🛠️ Steps in the Project
1. **Data Preprocessing** 📊
   - Menghapus nilai yang hilang (jika ada)
   - Mengonversi fitur kategorikal menggunakan **One-Hot Encoding**
   - Melakukan **Boolean Encoding** pada fitur biner (seperti `housing`, `loan`, `deposit`)

2. **Exploratory Data Analysis (EDA)** 🔍
   - Visualisasi distribusi data menggunakan **histogram** dan **boxplot**
   - Analisis outlier dengan **boxplot**
   
3. **Feature Engineering** ⚙️
   - Seleksi fitur yang paling relevan
   - Normalisasi atau standarisasi fitur (jika diperlukan)
   
4. **Model Training & Evaluation** 🤖
   - Melatih model **Random Forest** dan **XGBoost**
   - Evaluasi menggunakan **akurasi, precision, recall, dan F1-score**
   - Menampilkan **confusion matrix** untuk melihat performa model

5. **Model Comparison & Selection** 🏆
   - Membandingkan performa kedua model
   - Memilih model terbaik berdasarkan metrik evaluasi

## ✨ Unique Aspects
✅ Menggunakan **XGBoost**, salah satu model boosting terbaik untuk klasifikasi

✅ **One-Hot Encoding** dan **Boolean Encoding** untuk menangani data kategorikal

✅ **Visualisasi EDA** yang membantu memahami pola dalam data

✅ **Evaluasi mendalam** menggunakan berbagai metrik dan confusion matrix

## 📌 Conclusion
Proyek ini menunjukkan bahwa model **XGBoost** dan **Random Forest** dapat digunakan untuk memprediksi kemungkinan pelanggan melakukan deposit. Hasil evaluasi membantu memilih model terbaik yang dapat digunakan untuk implementasi lebih lanjut.

