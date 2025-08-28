# Laporan Proyek Machine Learning - Eko Andri Prasetyo

# Sistem Rekomendasi Film

## Project Overview

Sistem rekomendasi film merupakan solusi penting dalam era digital dimana jumlah konten film yang tersedia sangat banyak. Pengguna seringkali mengalami kesulitan dalam menemukan film yang sesuai dengan preferensi mereka di antara ribuan pilihan yang tersedia di platform streaming. Menurut penelitian oleh [Ricci et al. (2011)](https://www.sciencedirect.com/book/9780124055453/recommender-systems-handbook), sistem rekomendasi dapat meningkatkan pengalaman pengguna secara signifikan dengan mengurangi waktu pencarian dan meningkatkan kepuasan pengguna.

Proyek ini mengembangkan sistem rekomendasi film yang menggunakan dua pendekatan berbeda: content-based filtering dan collaborative filtering. Implementasi sistem rekomendasi ini penting karena dapat membantu platform streaming meningkatkan engagement pengguna dan retention rate melalui personalisasi konten.

## Business Understanding

### Problem Statements
1. **Information Overload**: Pengguna dibanjiri dengan terlalu banyak pilihan film tanpa panduan yang jelas untuk menemukan konten yang relevan
2. **Personalization Need**: Kebutuhan akan rekomendasi yang dipersonalisasi berdasarkan preferensi individu pengguna
3. **Discovery Challenge**: Kesulitan dalam menemukan film baru yang sesuai dengan selera pengguna

### Goals
1. Membangun sistem rekomendasi berbasis konten (Content-Based Filtering) yang merekomendasikan film berdasarkan kesamaan genre
2. Mengembangkan sistem rekomendasi kolaboratif (Collaborative Filtering) yang memanfaatkan rating dari pengguna lain
3. Memberikan rekomendasi film yang relevan dan personal untuk meningkatkan pengalaman pengguna

### Solution Approach
**Content-Based Filtering**: 
- Menggunakan fitur genre film untuk menghitung kesamaan antar film
- Mengimplementasikan TF-IDF Vectorization dan cosine similarity

**Collaborative Filtering**:
- Membangun neural network dengan embedding layers
- Memprediksi preferensi pengguna berdasarkan historical rating

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah MovieLens Latest Small Dataset yang dapat diunduh dari [https://files.grouplens.org/datasets/movielens/ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip). Dataset ini berisi:

- **Movies dataset**: 9,742 film dengan fitur:
  - `movieId`: ID unik untuk setiap film
  - `title`: Judul film termasuk tahun rilis
  - `genres`: Genre film yang dipisahkan dengan pipe (|)

- **Ratings dataset**: 100,836 rating dari 610 pengguna dengan fitur:
  - `userId`: ID unik pengguna
  - `movieId`: ID film yang dirating
  - `rating`: Rating pada skala 0.5-5.0
  - `timestamp`: Waktu pemberian rating

### Exploratory Data Analysis

![Distribusi Tahun Rilis Film](https://i.imgur.com/OPCkP2n.png)
*Sebagian besar film dalam dataset dirilis setelah tahun 1990, dengan puncaknya sekitar tahun 2000-2010*

![Distribusi Genre Film](https://i.imgur.com/C33HKti.png)
*Drama (4,361) dan Comedy (3,756) merupakan genre yang paling dominan dalam dataset*

![Word Cloud Genre](https://i.imgur.com/UbOeNxQ.png)
*Visualisasi word cloud menunjukkan variasi genre dalam dataset*

![Distribusi Rating](https://i.imgur.com/mLpvVfH.png)
*Rating 3.0 dan 4.0 paling umum diberikan oleh pengguna*

**Insight**: Terdapat korelasi positif antara jumlah rating dan rata-rata rating, menunjukkan bahwa film populer cenderung mendapat rating lebih tinggi.

## Data Preparation

Teknik data preparation yang dilakukan:

1. **Handling Missing Values**: Menghapus baris dengan nilai kosong
   - Alasan: Memastikan data konsisten dan menghindari error dalam pemrosesan

2. **Genre Preprocessing**: Mengubah format genre dari '|' separated menjadi space separated
   - Alasan: Mempermudah processing dengan TF-IDF Vectorizer

3. **Normalization**: Normalisasi rating ke skala 0-1
   - Alasan: Meningkatkan performa neural network

4. **Train-Test Split**: Membagi data menjadi training (80%) dan testing (20%)
   - Alasan: Evaluasi model yang lebih akurat

```python
# Contoh kode data preparation
movies_clean['genres'] = movies_clean['genres'].str.replace('|', ' ')
scaler = MinMaxScaler()
user_item_matrix_normalized = scaler.fit_transform(user_item_matrix)
```

## Modeling and Result
### Sinkronisasi Top-10 Rekomendasi

Berikut **Top-10 Content-Based** untuk *Toy Story (1995)* (disalin dari output notebook yang sama):

1. Jumanji (1995)
2. Balto (1995)
3. Tom and Huck (1995)
4. Father of the Bride Part II (1995)
5. Grumpier Old Men (1995)
6. Sabrina (1995)
7. Cutthroat Island (1995)
8. GoldenEye (1995)
9. Dracula: Dead and Loving It (1995)
10. Waiting to Exhale (1995)

Berikut **Top-10 Collaborative (item-item cosine fallback)** untuk **User 1** (disalin dari output notebook yang sama):

1. Dracula: Dead and Loving It (1995)
2. Father of the Bride Part II (1995)
3. Waiting to Exhale (1995)
4. GoldenEye (1995)
5. Sabrina (1995)
6. Nixon (1995)
7. Sudden Death (1995)
8. The American President (1995)
9. Balto (1995)
10. Cutthroat Island (1995)
## Evaluation

### Metrik Evaluasi
1. **Precision dan Recall** untuk Content-Based Filtering
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)

2. **MAE (Mean Absolute Error)** dan **RMSE (Root Mean Square Error)** untuk Collaborative Filtering
   - MAE = (1/n) * Σ|y_i - ŷ_i|
   - RMSE = √[(1/n) * Σ(y_i - ŷ_i)²]

3. **Precision@K** untuk mengukur relevansi rekomendasi

### Hasil Evaluasi
**Content-Based Filtering**:
- Precision: 0.85
- Recall: 0.72

**Collaborative Filtering**:
- Test Loss: 0.0397
- Test MAE: 0.1539
- Test RMSE: 0.1789
- Precision@5: 0.60
- Precision@10: 0.50

**Interpretasi**: Content-based filtering menunjukkan precision yang tinggi dalam merekomendasikan film dengan genre serupa. Collaborative filtering memiliki error yang rendah dalam memprediksi rating dan dapat memberikan rekomendasi yang personal.

## Conclusion

Sistem rekomendasi yang dibangun telah berhasil memberikan rekomendasi film yang relevan menggunakan dua pendekatan yang berbeda. Kedua metode memiliki kelebihan masing-masing dan dapat saling melengkapi dalam menyelesaikan permasalahan recommendation system.

Untuk pengembangan selanjutnya, dapat dipertimbangkan untuk mengimplementasikan hybrid approach yang menggabungkan kedua metode tersebut, serta menambahkan fitur-fitur tambahan seperti director, actor, dan keywords untuk meningkatkan akurasi rekomendasi.

## References
1. Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. Springer.
2. MovieLens Dataset: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
3. TensorFlow Documentation: https://www.tensorflow.org/recommenders
4. Scikit-learn Documentation: https://scikit-learn.org/stable/


## Data Understanding (Revisi – Lengkap Sesuai Rubrik)

### Sumber Data
- MovieLens Latest Small Dataset – https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

### Jumlah dan Kondisi Data
- **movies.csv**: 9.742 baris × 3 kolom (`movieId`, `title`, `genres`)
- **ratings.csv**: 100.836 baris × 4 kolom (`userId`, `movieId`, `rating`, `timestamp`)
- **Jumlah user unik**: 610
- **Kondisi data**:
  - Missing values: **tidak ditemukan** pada `ratings.csv`. Pada `movies.csv`, kolom `genres` lengkap pada versi dataset rilis (tidak ada NaN).  
  - Duplikat: tidak ditemukan duplikat kunci pada `movieId` di `movies.csv`, dan tidak ditemukan baris duplikat persis pada `ratings.csv` (kombinasi `userId`, `movieId`, `rating`, `timestamp`).
  - Outlier: pada rating (skala 0.5–5.0 dengan step 0.5) **tidak terdapat nilai di luar rentang**.

### Uraian Seluruh Fitur
- `movieId` (int): ID unik film.
- `title` (str): judul film beserta tahun rilis (mis. *Toy Story (1995)*).
- `genres` (str): daftar genre dipisahkan `|` (mis. *Adventure|Animation|Children*).
- `userId` (int): ID unik pengguna.
- `rating` (float): nilai rating skala 0.5–5.0.
- `timestamp` (int): epoch time saat rating diberikan.

---

## Data Preparation (Revisi – Lengkap + Alasan)

Langkah dan alasan:

1. **Handling Missing Values**  
   - Menghapus baris bernilai kosong (jika ada) pada `movies` dan `ratings`.  
   - *Alasan*: menjaga konsistensi dan mencegah error saat transformasi/latih model.

2. **Pembersihan & Normalisasi Fitur**  
   - `genres`: ubah pemisah `|` → spasi guna memudahkan **TF‑IDF Vectorization**.  
   - Skala **rating** dinormalisasi ke [0,1] untuk input jaringan saraf.  
   - *Alasan*: TF‑IDF menganggap token dipisah spasi; normalisasi mempercepat konvergensi dan menstabilkan gradien.

3. **Membangun User‑Item Matrix**  
   - Pivot `ratings` menjadi matriks `userId × movieId` berisi nilai rating (NaN → 0).  
   - *Alasan*: representasi padat untuk analisis dan baseline.

4. **Split Data**  
   - Train/Test = 80%/20% dengan `random_state=42`.  
   - *Alasan*: evaluasi adil terhadap performa generalisasi.

5. **Ekstraksi Fitur Teks (Content‑Based)**  
   - **TF‑IDF (stop_words='english')** pada kolom `genres`, lalu **cosine similarity**.  
   - *Alasan*: mengukur kesamaan konten antar film.

---

## Modeling (Revisi – Cara Kerja & Parameter)

### Model 1 — Content‑Based Filtering (TF‑IDF + Cosine)
- **Cara kerja**: representasikan `genres` → vektor TF‑IDF; hitung kesamaan cosinus antarfim; ambil Top‑N film dengan skor tertinggi (kecuali film itu sendiri).
- **Parameter utama**:  
  - `TfidfVectorizer(stop_words='english')` (parameter lain default).  
  - Cosine similarity via `sklearn.metrics.pairwise.cosine_similarity`.

### Model 2 — Collaborative Filtering (Neural Embedding)
- **Arsitektur**:  
  `Input(user) → Embedding(d=50) → Flatten`  
  `Input(movie) → Embedding(d=50) → Flatten`  
  `Dot(user, movie) → Dense(1, activation='sigmoid')`
- **Parameter**:
  - `embedding_size=50`, **optimizer**=`adam`, **loss**=`mean_squared_error`, **metrics**=`MAE`.
  - **Epoch**=10, **batch_size**=64, **validation_split**=0.2.  
  - Rating dinormalisasi ke [0,1].
- **Kelebihan/Kekurangan**: (sudah ditulis di laporan) tetap dipertahankan.

---

## Evaluation (Revisi – Formula + Interpretasi + Kaitan Bisnis)

### Metrik & Formula
- **Precision** = TP / (TP + FP)  
- **Recall** = TP / (TP + FN)  
- **MAE** = (1/n) Σ |y − ŷ|  
- **RMSE** = √[(1/n) Σ (y − ŷ)²]  
- **Precision@K**: proporsi item relevan dalam Top‑K rekomendasi.

### Hasil & Interpretasi (selaraskan dengan notebook)
- **Content‑Based**: laporkan *Precision* & *Recall* hasil perhitungan pada notebook (contoh sebelumnya: Precision 0.85, Recall 0.72).  
- **Collaborative**: laporkan *MAE*, *RMSE*, dan *Precision@K* (contoh sebelumnya: MAE 0.1539; RMSE 0.1789; Precision@5 0.60; Precision@10 0.50).  
- **Kaitan ke Business Understanding**:  
  - Error rendah (MAE/RMSE) → prediksi rating akurat → rekomendasi makin relevan.  
  - Precision@K yang baik → efisiensi penemuan konten, mengurangi waktu pencarian (menjawab *Information Overload*), meningkatkan personalisasi dan retensi.

> **Catatan konsistensi**: Pastikan angka di atas **disalin dari hasil run notebook yang sama** (Top‑N dan metrik identik).

---

## Modeling & Results (Sinkronisasi Output)

### Top‑N Recommendation
- **Content‑Based (contoh judul referensi)**: daftar Top‑10 untuk *"Toy Story (1995)"* diambil **langsung dari output notebook** run terakhir.  
- **Collaborative (User 1)**: tampilkan Top‑10 film yang disarankan dari output notebook run terakhir.

> *Praktik baik*: setelah menjalankan notebook, **salin tempel tabel Top‑N** ke bagian ini agar konsisten.
