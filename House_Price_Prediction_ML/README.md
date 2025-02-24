# 🏡 Prediksi Harga Rumah dengan XGBRegressor

## 📌 Gambaran Proyek
Proyek ini bertujuan untuk memprediksi harga rumah di California menggunakan model **XGBRegressor**. Dataset yang digunakan adalah **California Housing dataset** dari `sklearn.datasets`. 

💡 **Langkah-langkah utama dalam proyek ini:**
1. **Memuat Dataset**: Mengambil data dari `sklearn.datasets`.
2. **Eksplorasi Data**: Mengecek fitur, deskripsi, dan statistik data.
3. **Pra-pemrosesan**: Menangani nilai yang hilang, melihat korelasi antar fitur.
4. **Membagi Data**: Membagi dataset menjadi data latih dan uji.
5. **Melatih Model**: Menggunakan `XGBRegressor` untuk melatih model.
6. **Evaluasi Model**: Menggunakan R-Squared, Mean Squared Error, dan Mean Absolute Error.
7. **Visualisasi**: Membandingkan harga aktual dan prediksi dengan scatter plot.

---

## 📝 Implementasi Kode

### 1️⃣ Persiapan Data
🔹 Mengimpor pustaka Python yang diperlukan seperti `pandas`, `numpy`, dan `seaborn` untuk analisis data, serta `XGBRegressor` untuk model machine learning.
🔹 Mengambil dataset **California Housing** dari `sklearn.datasets` dan menyimpannya dalam **DataFrame** agar lebih mudah dianalisis.
🔹 Mengecek informasi dasar dataset seperti jumlah baris & kolom, statistik deskriptif, serta mengidentifikasi jika ada nilai yang hilang.

### 2️⃣ Eksplorasi Data
📊 **Analisis Korelasi**: Menggunakan heatmap untuk melihat hubungan antara fitur dalam dataset. 
💡 **Mengapa penting?** Ini membantu kita memahami fitur mana yang paling berpengaruh terhadap harga rumah.

### 3️⃣ Membagi Data
🔀 **Menggunakan `train_test_split`** untuk membagi dataset menjadi data latih dan uji dengan perbandingan 80:20. 
🃏 **Analogi:** Seperti membagi kartu dalam permainan poker—kita memastikan sebagian data digunakan untuk belajar (latih), sementara sebagian lain disimpan untuk menguji keakuratan model.

### 4️⃣ Melatih Model
🤖 **Menggunakan `XGBRegressor`**: Algoritma machine learning yang kuat untuk prediksi harga rumah. 
🚀 **Keunggulan XGBoost**: Cepat, efisien, dan mampu menangani dataset besar dengan baik.

### 5️⃣ Evaluasi Model
📏 **Metode evaluasi**: 
- **R-Squared (R²)**: Seberapa baik model menjelaskan variasi dalam data.
- **Mean Squared Error (MSE)**: Rata-rata dari kuadrat kesalahan prediksi.
- **Mean Absolute Error (MAE)**: Rata-rata selisih absolut antara nilai prediksi dan aktual.

### 6️⃣ Visualisasi Hasil
📌 **Scatter Plot**: Membandingkan harga aktual vs harga prediksi menggunakan grafik.
📉 **Mengapa penting?** Jika titik-titik sejajar dengan garis diagonal, berarti prediksi model cukup akurat!

---

## 🎭 Analogi untuk Memahami `random_state`

Bayangkan `random_state` sebagai **aturan dalam mengocok kartu**:
- Jika `random_state=2`, kartu akan selalu dikocok dengan cara yang sama, sehingga pemain selalu mendapatkan kartu yang sama di setiap permainan.
- Jika tidak disetel, maka di setiap permainan kartu akan dikocok secara acak dengan hasil yang berbeda.

🃏 Ini memastikan bahwa pembagian data latih dan uji selalu sama setiap kali kode dijalankan.

## ✅ Kesimpulan
Proyek ini berhasil memprediksi harga rumah menggunakan `XGBRegressor`, menunjukkan korelasi kuat antara nilai aktual dan prediksi. Evaluasi dilakukan menggunakan **R-Squared, MSE, dan MAE**. Model dapat ditingkatkan lebih lanjut dengan **tuning hyperparameter atau feature engineering**.