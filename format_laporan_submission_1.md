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
Lampiran Grafik EDA:
![Screenshot 2025-05-26 223042](https://github.com/user-attachments/assets/13f6f57b-e23b-4ca4-9342-feccb2b68fec)

* **Tidak ada nilai kosong (missing values)** pada sebagian besar kolom berdasarkan hasil `df.isnull().sum()`. Namun, kolom `TotalCharges` bertipe `object` meskipun berisi angka. Setelah dikonversi ke `float`, ditemukan **11 nilai NaN**, yang akan ditangani pada tahap data preparation.
* **Tidak terdapat duplikat** dalam data.
* Kolom `customerID` merupakan identifier unik yang tidak memiliki kontribusi terhadap prediksi dan akan dihapus.
* **Distribusi target (`Churn`) menunjukkan ketidakseimbangan kelas**:

  * Sekitar **73%** data berada pada kelas `No` (tidak churn)
  * Sekitar **27%** data berada pada kelas `Yes` (churn)
    Hal ini menandakan bahwa dataset bersifat **imbalanced**, yang perlu menjadi perhatian khusus dalam proses pelatihan model.

* Beberapa fitur seperti `Contract`, `PaymentMethod`, `InternetService`, dan `OnlineBackup` memiliki **kategori yang berbeda-beda**, yang perlu diubah menjadi representasi numerik menggunakan encoding pada tahap selanjutnya.

### **Correlation Matrix**
Lampiran Correlation Matrix:
![image](https://github.com/user-attachments/assets/f2bc62dd-e116-42c6-8245-8ab284a64263)

* Setelah proses encoding, dilakukan analisis **matriks korelasi (correlation matrix)** untuk mengetahui hubungan antar fitur. Visualisasi seperti heatmap digunakan untuk mengidentifikasi fitur-fitur yang paling berpengaruh, serta untuk mendeteksi multikolinearitas antar fitur.

Berikut adalah detail dari korelasi antar fitur:
- `gender`: Tingkat korelasi rendah dengan semua fitur
- `SeniorCitizen`: Tingkat korelasi rendah dengan semua fitur
- `Partner`: Tingkat korelasi sebesar 0.45 dengan fitur 'Dependents'
- `Dependents`: Tingkat korelasi sebesar 0.45 dengan fitur 'Partner'
- `PhoneService`: Tingkat korelasi sebesar 0.68 dengan fitur 'MultipleLines'
- `MultipleLines`: Tingkat korelasi sebesar 0.68 dengan fitur 'PhoneService'
- `InternetService`: Tingkat korelasi sebesar 0.72 dengan fitur 'TechSupport' dan 'OnlineSecurity'
- `OnlineSecurity`: Tingkat korelasi sebesar 0.74 dengan fitur 'TechSupport'
- `OnlineBackup`: Tingkat korelasi sebesar 0.71 dengan fitur 'TechSupport', 'DeviceProtection', 'MonthlyCharges' dan 'OnlineSecurity'
- `DeviceProtection`: Tingkat korelasi sebesar 0.75 dengan fitur 'StreamingTV' dan 'StreamingFilm'
- `TechSupport`: Tingkat korelasi sebesar 0.74 dengan fitur 'OnlineSecurity'
- `StreamingTV` & `StreamingMovies`: Tingkat korelasi sebesar 0.82 dengan fitur 'MonthlyCharges'
- `Contract`: Tingkat korelasi sebesar 0.67 dengan fitur 'tenure'
- `PaymentMethod`: Tingkat korelasi sebesar 0.35 dengan fitur 'Contract'
- `PaperlessBilling`: Tingkat korelasi sebesar 0.35 dengan fitur 'MonthlyCharges'
- `tenure`: Tingkat korelasi sebesar 0.83 dengan fitur 'TotalCharges'
- `MonthlyCharges`: Tingkat korelasi sebesar 0.82 dengan fitur `StreamingTV` & `StreamingMovies`
- `TotalCharges`: Tingkat korelasi sebesar 0.83 dengan fitur 'tenure'
- `Churn`: Tingkat korelasi sebesar 0.19 dengan fitur 'MonthlyCharges' dan 'PaperlessBilling'


## Data Preparation
Tahap ini mencakup proses transformasi data mentah menjadi bentuk yang siap digunakan untuk pelatihan model machine learning. Berikut adalah langkah-langkah yang dilakukan:

---

###  1. **Transform Tipe Data**

* Dilakukan konversi kolom `TotalCharges` dari tipe `object` ke `float` menggunakan `pd.to_numeric(errors='coerce')`.

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```

---

###  2. **Handling Missing Values**

* Awalnya, dataset tidak memiliki missing value eksplisit (`df.isnull().sum()` menunjukkan 0 pada semua kolom).
* Namun, setelah dilakukan konversi kolom `TotalCharges`, sebanyak **11 nilai menjadi `NaN`** karena berisi spasi atau string kosong.
* Missing value ini kemudian ditangani dengan cara **mengisi nilai yang hilang menggunakan median** dari kolom `TotalCharges`. Penggunaan median dipilih karena lebih tahan terhadap outlier dibandingkan mean.

```python
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
```

---

###  3. **Drop Kolom Tidak Relevan**

* Kolom `customerID` dihapus karena merupakan **identifier unik** yang tidak memiliki nilai prediktif dalam klasifikasi churn.

```python
df.drop(columns=['customerID'], inplace=True)
```

---

###  4. **Encoding Kategorikal**

* Untuk dapat digunakan dalam pemodelan, fitur kategorikal perlu diubah ke format numerik.
* Encoding dilakukan dengan pendekatan **label encoding**, mengubah kategori ke nilai 0/1/2/dst. secara eksplisit, terutama untuk fitur-fitur biner seperti `gender`, `Partner`, `Dependents`, `PaperlessBilling`, dan lainnya.

```python
# Encode fitur kategorikal
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 2, 'No': 1, 'No phone service': 0})
df['InternetService'] = df['InternetService'].map({'DSL': 2, 'Fiber optic': 1, 'No': 0})
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['TechSupport'] = df['TechSupport'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['StreamingTV'] = df['StreamingTV'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['Contract'] = df['Contract'].map({'Two year': 2, 'One year': 1, 'Month-to-month': 0})
df['PaymentMethod'] = df['PaymentMethod'].map({'Bank transfer (automatic)':3, 'Credit card (automatic)': 2, 'Mailed check': 1, 'Electronic check': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
```

---

###  5. **Feature Selection dan Final Dataset**

* Dataset akhir yang digunakan untuk pelatihan model terdiri atas kombinasi fitur numerik dan hasil encoding fitur kategorikal.
* Kolom target yang diprediksi adalah `Churn`.

---

### 6. Pembagian Dataset (Train-Test Split)

* Dataset dibagi menjadi data **pelatihan (training)** dan **pengujian (testing)** menggunakan fungsi `train_test_split` dari `sklearn.model_selection`.
* **Proporsi pembagian** adalah 80% data untuk pelatihan dan 20% untuk pengujian.
* Parameter `random_state=42` digunakan agar proses pembagian data bersifat **reproducible** (hasil selalu konsisten setiap dijalankan).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---



## Modeling

Model yang digunakan dalam proyek ini adalah **Random Forest Classifier**, yaitu algoritma ensemble berbasis decision tree yang sangat populer dalam klasifikasi dan regresi.

###  Cara Kerja Algoritma Random Forest:

Random Forest bekerja dengan membangun **banyak pohon keputusan (decision trees)** pada subset acak dari data pelatihan, lalu menggabungkan hasil prediksi dari masing-masing pohon (melalui voting mayoritas untuk klasifikasi) untuk menghasilkan prediksi akhir. Proses ini dikenal sebagai **bagging (Bootstrap Aggregating)**, di mana setiap pohon:

* Dilatih dengan data yang di-*sampling* secara acak dengan pengembalian (bootstrapping).
* Pada setiap node, hanya subset acak dari fitur yang dipertimbangkan untuk split, sehingga meningkatkan keberagaman antar pohon.

**Keuntungan utama** Random Forest adalah:

* Lebih tahan terhadap **overfitting** dibanding single decision tree.
* Mampu menangani fitur kategorikal dan numerik secara bersamaan.
* Memberikan estimasi **feature importance**, yang berguna untuk interpretasi model.

---

###  Parameter Model yang Digunakan

Model dibangun menggunakan **`RandomForestClassifier` dari library `sklearn.ensemble`** dengan parameter sebagai berikut:

* `random_state=42`: Digunakan untuk memastikan hasil model dapat **direproduksi**. Nilai 42 dipilih secara arbitrer namun umum digunakan sebagai nilai acuan.
* Parameter lainnya seperti `n_estimators`, `max_depth`, dan sebagainya menggunakan nilai **default** dari scikit-learn.

```python
model = RandomForestClassifier(random_state=42)
```

Model dilatih pada data pelatihan (`X_train`, `y_train`) menggunakan:

```python
model.fit(X_train, y_train)
```

Setelah pelatihan, model digunakan untuk memprediksi data uji (`X_test`) dan dilakukan evaluasi performa menggunakan metrik klasifikasi seperti **accuracy**, **precision**, **recall**, dan **f1-score**.

###  Feature Importance:
Lampiran Grafik Feature Importance:
![image](https://github.com/user-attachments/assets/8cb02cfd-f076-4558-8e77-7eafa7830107)
* Setelah model dilatih, kita dapat mengevaluasi **seberapa besar kontribusi masing-masing fitur** terhadap prediksi churn dengan melihat `feature_importances_`.
* Fitur dengan nilai importance lebih tinggi memiliki **pengaruh lebih besar** terhadap keputusan model.
* Hasil ini dirangkum ke dalam dataframe dan diurutkan menurun, kemudian divisualisasikan dalam bentuk horizontal bar chart.
* Fitur seperti `TotalCharges`, `PaymentMethod`, `Contract`, `tenure`, dan `MonthlyCharges` berada di posisi atas, menunjukkan bahwa:

  * **Biaya total (TotalCharges)** dan **Biaya bulanan (MonthlyCharges)** yang tinggi bisa menjadi faktor risiko churn.
  * **Jenis kontrak** pelanggan (bulanan, tahunan) sangat memengaruhi kemungkinan churn.
  * **Lama berlangganan (tenure)** umumnya memiliki korelasi negatif dengan churn (semakin lama, semakin kecil kemungkinan churn).
  * **Metode pembayaran (PaymentMethod)** model prediksi churn karena metode pembayaran pelanggan dapat merefleksikan preferensi, kebiasaan, dan kemungkinan loyalitas mereka terhadap layanan.

## Evaluation
### A. Metrik Evaluasi yang Digunakan
Dalam proyek prediksi customer churn ini, kami menggunakan empat metrik evaluasi utama untuk mengukur performa model:

1. **Precision**  
   - *Definisi*: Rasio prediksi positif yang benar (True Positive) terhadap seluruh prediksi positif (True Positive + False Positive).  
   - *Relevansi*: Mengukur seberapa akurat model dalam memprediksi churn (menghindari false alarm).  

2. **Recall (Sensitivity)**  
   - *Definisi*: Rasio prediksi positif yang benar (True Positive) terhadap seluruh kasus aktual positif (True Positive + False Negative).  
   - *Relevansi*: Mengukur kemampuan model menemukan pelanggan yang benar-benar berisiko churn (menghindari missed detection).  

3. **F1-Score**  
   - *Definisi*: Rata-rata harmonik (harmonic mean) dari precision dan recall.  
   - *Relevansi*: Memberikan balance antara precision dan recall, terutama penting untuk data tidak seimbang.  

4. **Accuracy**  
   - *Definisi*: Rasio prediksi benar (True Positive + True Negative) terhadap total sampel.  
   - *Relevansi*: Mengukur performa keseluruhan model, tetapi kurang informatif untuk data imbalance.  



### B. Analisis Hasil Evaluasi

| Metrik    | Kelas 0 (No Churn) | Kelas 1 (Churn) |
| --------- | ------------------ | --------------- |
| Precision | 0.83               | 0.66            |
| Recall    | 0.91               | 0.47            |
| F1-Score  | 0.87               | 0.55            |
| Support   | 1036               | 373             |

### C. Interpretasi:  
####  **Kelas 0 (Tidak Churn):**

* **Precision 0.83:** Dari seluruh prediksi "tidak churn", sebanyak 83% benar.
* **Recall 0.91:** Model berhasil menangkap 91% pelanggan yang memang tidak churn.
* **F1-score 0.87:** Performa keseluruhan untuk kelas mayoritas sangat baik.
* Ini wajar karena kelas 0 adalah **kelas dominan (sekitar 73%)**, sehingga model memiliki cukup data untuk belajar mengenalinya.

####  **Kelas 1 (Churn):**

* **Precision 0.66:** Dari semua prediksi "churn", hanya 66% yang benar.
* **Recall 0.47:** Model hanya berhasil menangkap 47% pelanggan yang benar-benar churn.
* **F1-score 0.55:** Menunjukkan bahwa model masih **kesulitan mengenali kelas minoritas (churn)** secara akurat.

####  **Akurasi dan Rata-Rata:**
* **Accuracy 0.79:** Model secara keseluruhan benar memprediksi 79% data.
* **Macro average:**

  * Rata-rata dari masing-masing kelas tanpa mempertimbangkan proporsi data.
  * **Recall macro hanya 0.69**, menunjukkan bahwa **kelas minoritas tidak dipelajari dengan baik**.
* **Weighted average:**

  * Rata-rata yang mempertimbangkan jumlah sampel per kelas.
  * Nilainya lebih tinggi karena dominasi kelas 0.


D. Rekomendasi Perbaikan
1. **Handling Class Imbalance**  
   - Gunakan teknik oversampling (SMOTE) atau class weighting (`class_weight='balanced'`).  
2. **Optimasi Threshold**  
   - Turunkan threshold prediksi churn untuk meningkatkan recall (misalnya, dari 0.5 ke 0.3).  
3. **Eksperimen Model Lain**  
   - Coba algoritma seperti **XGBoost** atau **LightGBM** yang lebih robust terhadap imbalance.  
4. **Feature Engineering**  
   - Tambahkan fitur interaksi (contoh: `MonthlyCharges/tenure`) untuk meningkatkan sinyal churn.  

E. Kesimpulan:  
Model saat ini cukup baik dalam memprediksi "No Churn" tetapi kurang optimal untuk deteksi dini churn. Fokus perbaikan harus pada peningkatan recall kelas 1 tanpa mengorbankan precision secara signifikan.

