# 🎬 Movie Recommendation System

## 📌 Overview
Proyek ini adalah sistem rekomendasi film berbasis **Content-Based Filtering** (Berbasis kesamaan teks dan Fitur Film)' menggunakan **TF-IDF** dan **cosine similarity**.
Aplikasi ini memungkinkan pengguna untuk memasukkan judul film dan menerima rekomendasi film serupa berdasarkan genre dan ringkasan cerita.

## 🤖 Kenapa Tidak Menggunakan Model Machine Learning?
1. **Rekomendasi berbasis teks** lebih cocok menggunakan teknik **Information Retrieval** dibanding supervised learning.
2. Tidak ada kebutuhan untuk melatih model, karena sistem hanya mengukur kesamaan teks antar film.
3. Machine Learning tradisional membutuhkan data yang terlabel, sedangkan sistem ini hanya menggunakan informasi deskriptif.

## 🔠 Kenapa Tidak Menggunakan Encoding?
Encoding (seperti One-Hot Encoding atau Label Encoding) digunakan untuk variabel kategori.
Dalam proyek ini, kita mengolah teks secara langsung menggunakan **TF-IDF**, sehingga tidak memerlukan encoding.

## 📊 Dataset
Dataset mengandung informasi film termasuk:
- **title**: Judul film
- **genre**: Genre film
- **overview**: Ringkasan film
- **vote_average**: Rating film

## ⚙️ Cara Kerja
1. **Menggabungkan genre dan overview** menjadi satu fitur teks.
2. **Menggunakan TF-IDF Vectorizer** untuk mengubah teks menjadi vektor numerik.
3. **Menghitung cosine similarity** antar film berdasarkan vektor TF-IDF.
4. **Menggunakan widget input dan slider** untuk memilih film dan mengatur minimal rating.
5. **Menampilkan daftar rekomendasi film** berdasarkan kesamaan teks dan rating minimal.

## 📜 Penjelasan Kode
1. **Memuat dataset**
   ```python
   df = pd.read_csv("/mnt/data/dataset.csv")
   df = df.dropna(subset=['overview', 'genre', 'vote_average'])
   ```
   - Menghapus data yang memiliki nilai kosong pada kolom penting.

2. **Menggabungkan fitur teks**
   ```python
   df['combined_features'] = df['genre'] + ' ' + df['overview']
   ```
   - Menggabungkan genre dan overview sebagai fitur untuk analisis kesamaan.

3. **Membuat TF-IDF dan Cosine Similarity**
   ```python
   tfidf = TfidfVectorizer(stop_words='english')
   tfidf_matrix = tfidf.fit_transform(df['combined_features'])
   cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
   ```
   - **TF-IDF** mengubah teks menjadi representasi numerik.
   - **Cosine similarity** mengukur kesamaan antar film.

4. **Membuat widget input dan slider**
   ```python
   movie_input = widgets.Text(placeholder='Masukkan judul film', description='Film:')
   rating_slider = widgets.FloatSlider(value=7.0, min=0, max=10, step=0.1, description='Min Rating:')
   ```
   - **Textbox** untuk memasukkan judul film.
   - **Slider** untuk memilih rating minimal film yang direkomendasikan.

5. **Fungsi rekomendasi**
   ```python
   def get_recommendations():
       title = movie_input.value.lower()
       min_rating = rating_slider.value
       idx = df[df['title'].str.lower() == title].index
       if idx.empty:
           with output:
               output.clear_output()
               print("Film tidak ditemukan. Coba judul lain.")
       else:
           idx = idx[0]
           sim_scores = list(enumerate(cosine_sim[idx]))
           sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
           movie_indices = [i[0] for i in sim_scores]
           recommended_movies = df.iloc[movie_indices]
           recommended_movies = recommended_movies[recommended_movies['vote_average'] >= min_rating]
           with output:
               output.clear_output()
               if recommended_movies.empty:
                   print("Tidak ada film yang sesuai dengan kriteria.")
               else:
                   display(recommended_movies[['title', 'genre', 'vote_average']])
   ```
   - Mencari indeks film berdasarkan judul input.
   - Mengurutkan film berdasarkan kemiripan.
   - Memfilter berdasarkan rating minimal.
   - Menampilkan rekomendasi jika ada, atau menampilkan pesan jika tidak ada.

6. **Menjalankan aplikasi dengan widget interaktif**
   ```python
   movie_input.observe(lambda change: get_recommendations(), names='value')
   rating_slider.observe(lambda change: get_recommendations(), names='value')
   display(movie_input, rating_slider, output)
   ```
   - **Menjalankan fungsi rekomendasi saat input berubah**.
   - **Menampilkan input, slider, dan output ke pengguna**.

## 🏆 Kesimpulan
Sistem rekomendasi ini bekerja dengan **mencari kesamaan antar film berdasarkan teks** dan **tidak memerlukan training model**. TF-IDF dan cosine similarity adalah solusi yang sederhana tetapi efektif untuk sistem berbasis konten.

