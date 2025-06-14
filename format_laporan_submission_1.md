# Laporan Proyek Sistem Rekomendasi Film - Berlian Ishma Zhafira Sujana

## Project Overview

Sistem rekomendasi menjadi komponen penting dalam industri digital saat ini, terutama dalam platform hiburan seperti Netflix, Amazon Prime, dan Disney+. Rekomendasi yang baik dapat meningkatkan kepuasan pengguna, memperpanjang waktu interaksi, dan meningkatkan retensi pelanggan.

Dalam proyek ini, kita membangun sistem rekomendasi film menggunakan pendekatan Collaborative Filtering berdasarkan MovieLens 25M dataset. Proyek ini bertujuan membantu pengguna menemukan film yang sesuai dengan preferensi mereka berdasarkan rating pengguna lain.

---

## Business Understanding

### Problem Statements

Bagaimana cara merekomendasikan film yang relevan kepada pengguna berdasarkan preferensi pengguna lain?

---

### Goals

Membuat sistem rekomendasi film yang mampu menghasilkan rekomendasi akurat bagi setiap pengguna.

---

### Solution Statements

1. Content-Based Filtering: Rekomendasi berdasarkan kesamaan konten film (belum diimplementasikan penuh di notebook, perlu ditambahkan untuk memenuhi kriteria tambahan).

2. Collaborative Filtering (SVD): Menggunakan pendekatan matrix factorization, khususnya TruncatedSVD.

---


## Data Understanding
Sumber Dataset: [MovieLens 25M Dataset] - (https://www.kaggle.com/datasets/garymk/movielens-25m-dataset)

Dataset berisi 25 juta rating dari 162,000 pengguna terhadap 62,000 film. Terdapat file Dataset terdiri dari genome-scores.csv, links.csv, ratings.csv, tags.csv, genome-tags.csv, movies.csv, README.txt. Fokus utama project adalah pada ratings.csv dan movies.csv., yang mencakup berbagai aspek pelanggan seperti:

Untuk movies.csv terdiri dari **62423 baris dan 3 kolom**, yang mencakup berbagai aspek seperti:

* **movieId**: pengidentifikasi film
* **title**: judul
* **genres**: genre dari film

Untuk ratings.csv terdiri dari **25000095 baris dan 4 kolom**, yang mencakup berbagai aspek seperti:

* **userId**: pengidentifikasi pengguna
* **movieId**: pengidentifikasi film
* **rating**: ulasan pengguna (rentang 0,5 bintang - 5,0 bintang)
* **timestamp**: mewakili detik sejak tengah malam Waktu Universal Terkoordinasi (UTC) tanggal 1 Januari 1970.

###  Exploratory Data Analysis (EDA):

* Dataset ratings yang awalnya terdiri dari **25000095 baris**, dikurangi hingga **135714  baris** karena RAM tidak menyanggupi dataset yang terlalu banyak. Pengurangan baris didasarkan oleh 1000 pengguna pertama (berdasarkan userId) dan hanya mengambil film yang setidaknya memiliki penilaian sebanyak 3.

* Dataset movies yang awalnya terdiri dari **62423 baris**, dikurangi hingga **5699 baris** karena RAM tidak menyanggupi dataset yang terlalu banyak. Pengurangan baris didasarkan oleh movieId hasil saring ratings_reduced.csv

* **Tidak ditemukan nilai duplikat** dalam dataset. Hal ini penting untuk memastikan tidak ada baris data yang secara tidak sengaja menggandakan informasi pelanggan.

* Untuk memahami keragaman nilai dalam fitur kategorikal seperti `rating` dilakukan eksplorasi menggunakan fungsi `value_counts()` agar mengetahui distribusi tiap jumlah rating.

* Dari hasil visualisasi dataset, didapatkan bahwa:

  * ![Grafik 10 Film dengan Rating 5 Terbanyak](https://github.com/user-attachments/assets/c3b9429d-f30c-4ee4-ad7c-fe1f6c465e93)
   Film dengan judul "The Shawshank Redemption" menjadi film dengan rating 5 terbanyak. Disusul oleh "Pulp Fiction", lalu "Schindler's List".

  * ![Grafik 10 Film dengan Rating Terbanyak](https://github.com/user-attachments/assets/8a722845-c207-4422-a590-26f4a794bac6)
   Film dengan judul "Forrest Gump" menjadi film dengan rating terbanyak. Disusul oleh "Pulp Fiction", lalu "The Shawshank Redemption".

  * ![Grafik Distribusi Rating Pengguna](https://github.com/user-attachments/assets/aff503bb-4e00-4209-a820-f15e50ab29f6)
   Rating 4 mendominasi angka rating lainnya. Disusul oleh angka 3, lalu angka 5.


## Data Preparation
Tahap ini mencakup proses transformasi data mentah menjadi bentuk yang siap digunakan untuk pelatihan model machine learning. Berikut adalah langkah-langkah yang dilakukan:

---

###  1. **Menggabungkan Data**:
   - Menggabungkan dataframe `ratings_reduced` (berisi data rating) dengan `movies_reduced` (berisi data film)
   - Digabungkan berdasarkan kolom `movieId` yang sama di kedua dataframe
   - Hasilnya adalah dataframe baru yang berisi informasi rating + detail film

```python
movie_data = pd.merge(ratings_reduced, movies_reduced, on='movieId')
```

---

###  2. **Membuat Matriks User-Item**:
   - Membuat matriks dengan:
     - Baris: `userId` (setiap pengguna)
     - Kolom: `title` (judul film)
     - Nilai: `rating` yang diberikan pengguna pada film tersebut
   - `fillna(0)` mengisi nilai kosong dengan 0 (artinya pengguna belum memberi rating)

```python
user_item_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
```

---

###  3. **Konversi ke Numpy Array**

   Konversi DataFrame menjadi array (matriks angka) supaya bisa diproses oleh model machine learning. Data ini berisi rating dari user terhadap film

```python
X = user_item_matrix.values
```

---

###  4. **Split Data**
   Data dibagi menjadi dua bagian:
   * **80%** untuk melatih model (X\_train)
   * **20%** untuk menguji model (X\_test)
     Ini bertujuan untuk melihat apakah model bisa bekerja dengan baik pada data baru yang belum pernah dilihat.

```python
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

---

## Modeling and Result

Collaborative Filtering adalah pendekatan rekomendasi berdasarkan pola interaksi antar pengguna, tanpa memerlukan informasi detail mengenai film itu sendiri. Sistem ini mengasumsikan bahwa jika dua pengguna menyukai item yang sama, maka mereka kemungkinan akan menyukai item lain yang sama juga.

Untuk implementasinya, kita menggunakan teknik Matrix Factorization dengan algoritma Singular Value Decomposition (SVD).

###  1. **Latih Model SVD**

   Di sini kita melatih model **SVD (Singular Value Decomposition)**.
   `n_components=50` berarti menyederhanakan data ke dalam 50 fitur utama — seperti merangkum informasi penting dari data.
   
```python
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(X_train)
```

---
### 2. **fungsi get_recommendations()**

Fungsi ini digunakan untuk **memberikan saran film** kepada user tertentu berdasarkan data dan model SVD yang telah kamu latih sebelumnya.

a. **Prediksi Rating yang Belum Ada**

* **`svd.transform()`**: mengubah data user ke bentuk fitur yang telah dipelajari model.
* **`svd.inverse_transform()`**: mengembalikan prediksi rating untuk semua film.

Artinya, kita meminta model untuk menebak apabila user ini menonton semua film, kira-kira rating-nya berapa?

b. **Buat Series dari Prediksi**

  Mengubah hasil prediksi jadi daftar berisi:

  * Nama film
  * Prediksi rating dari user

c. **Buang Film yang Sudah Ditonton**

  Karena kita **tidak ingin menyarankan film yang sudah ditonton**, bagian ini akan **membuang** semua film yang user sudah beri rating.

  Lalu, film sisanya **diurutkan dari yang prediksinya paling tinggi ke rendah**.

```python
def get_recommendations(user_id, n_recommendations=5):
    # Get user ratings
    user_ratings = user_item_matrix.loc[user_id].values.reshape(1, -1)

    # Prediksi rating
    pred_ratings = svd.inverse_transform(svd.transform(user_ratings))
    pred_ratings = pd.Series(pred_ratings[0], index=user_item_matrix.columns)

    # Filter film yang belum ditonton
    watched_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = pred_ratings.drop(watched_movies).sort_values(ascending=False)

    return recommendations.head(n_recommendations)
```

### 3. Hasil

Hasil akhirnya berupa daftar film yang **paling direkomendasikan** oleh model untuk user tersebut.

Hasil untuk User 1:

```
Rekomendasi untuk User 1:
Memento (2000)
Fight Club (1999)
The Lord of the Rings: The Return of the King (2003)
The Lord of the Rings: The Fellowship of the Ring (2001)
Kill Bill: Vol. 1 (2003)
```

**Artinya:**

* Model memprediksi bahwa jika **User 1** menonton film-film di atas, **dia kemungkinan besar akan memberi rating tinggi**. Karena itu, film-film di atas **layak direkomendasikan**.
* Nilai prediksi bisa lebih dari 1 karena SVD tidak dibatasi oleh skala rating awal (misalnya 0–5).
* Semakin tinggi angka `pred_ratings`, semakin besar kemungkinan user akan menyukai film itu, dan angka tertinggi dipegang oleh film dengan judul "Memento".

---

## Evaluasi

### 1. **Prediksi dan Hitung Error (RMSE)**

* **RMSE (Root Mean Squared Error)** mengukur seberapa besar **rata-rata kesalahan** model dalam membuat prediksi.
* Nilai RMSE ini **berkisar dari 0 ke atas**. Semakin kecil, semakin **akurat** modelnya.

```python
X_pred = svd.inverse_transform(svd.transform(X_test))
rmse = np.sqrt(mean_squared_error(X_test, X_pred))
print(f"RMSE: {rmse:.4f}")

#Hasil RMSE: 0.4388
```

* Nilai **0.4388** berarti model cukup **baik** dalam merekomendasikan item, karena kesalahan rata-rata per prediksi cukup kecil (kurang dari 0.5).
* RMSE di bawah **0.5** cukup bagus, terutama jika skalanya dari 0 sampai 1 atau 5.

### 2. **Visualisasi Variance dari Komponen SVD**
![Evaluasi Model](https://github.com/user-attachments/assets/fa7c4bc3-de49-4830-adaa-f8a23ea4eed8)
Ini menunjukkan bahwa:

   * Komponen pertama (awal-awal) **menyumbang banyak informasi**.
   * Semakin banyak komponen ditambahkan, **tambahan informasi yang diperoleh semakin sedikit** (diminishing returns).
   * Dari grafik, terlihat bahwa dengan **50 komponen**, kita hanya bisa menjelaskan sekitar **49-50%** dari total informasi dalam data asli.
   * Ini normal untuk data yang sangat sparse (jarang terisi), seperti yang kita miliki dengan **sparsity 97.62%**.


## Kesimpulan:  
  * **SVD cocok digunakan** karena mampu menangkap pola penting meskipun data sangat kosong.

  * Sparsity tinggi (97.62%) Data sangat jarang diisi rating — cocok pakai model seperti SVD yang bisa memprediksi data hilang.                                 
  * SVD Explained Variance membantu memilih berapa banyak "fitur tersembunyi" (komponen) yang perlu digunakan untuk mendapatkan representasi data yang bagus.

