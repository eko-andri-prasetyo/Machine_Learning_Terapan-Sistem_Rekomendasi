# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Project Overview

Sistem rekomendasi film merupakan solusi penting dalam era digital dimana jumlah konten film yang tersedia sangat banyak. Pengguna seringkali mengalami kesulitan dalam menemukan film yang sesuai dengan preferensi mereka di antara ribuan pilihan yang tersedia di platform streaming. Menurut penelitian oleh Ricci et al. (2011), sistem rekomendasi dapat meningkatkan pengalaman pengguna secara signifikan dengan mengurangi waktu pencarian dan meningkatkan kepuasan pengguna.

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

Dataset yang digunakan dalam proyek ini adalah MovieLens Latest Small Dataset yang dapat diunduh dari https://files.grouplens.org/datasets/movielens/ml-latest-small.zip. Dataset ini berisi informasi tentang film, rating pengguna, dan tag.

### Movies Dataset
**Jumlah Data**: 9,742 baris × 3 kolom

**Kondisi Data**:
- Tidak terdapat missing values pada kolom movieId, title, dan genres
- Terdapat 13 missing values pada kolom year yang diekstrak dari title

**Fitur**:
- `movieId`: ID unik untuk setiap film (integer)
- `title`: Judul film termasuk tahun rilis (string)
- `genres`: Genre film yang dipisahkan dengan pipe (|) (string)

### Ratings Dataset  
**Jumlah Data**: 100,836 baris × 4 kolom

**Kondisi Data**:
- Tidak terdapat missing values
- Rating berkisar dari 0.5 hingga 5.0 dengan interval 0.5

**Fitur**:
- `userId`: ID unik pengguna (integer)
- `movieId`: ID film yang dirating (integer) 
- `rating`: Rating pada skala 0.5-5.0 (float)
- `timestamp`: Waktu pemberian rating dalam format UNIX timestamp (integer)

### Eksplorasi Data
Analisis menunjukkan bahwa genre Drama (4,361) dan Comedy (3,756) merupakan genre yang paling dominan dalam dataset. Distribusi tahun rilis menunjukkan sebagian besar film dirilis setelah tahun 1990, dengan puncaknya sekitar tahun 2000-2010. Rating 3.0 dan 4.0 paling umum diberikan oleh pengguna.

## Data Preparation

Tahapan preprocessing data yang dilakukan:

### 1. Handling Missing Values
Menghapus baris dengan nilai kosong untuk memastikan data konsisten dan menghindari error dalam pemrosesan.

```python
movies_clean = movies.dropna()
ratings_clean = ratings.dropna()
```

### 2. Genre Preprocessing  
Mengubah format genre dari '|' separated menjadi space separated untuk mempermudah processing dengan TF-IDF Vectorizer.

```python
movies_clean['genres'] = movies_clean['genres'].str.replace('|', ' ')
```

### 3. Normalization
Normalisasi rating ke skala 0-1 untuk meningkatkan performa neural network.

```python
scaler = MinMaxScaler()
user_item_matrix_normalized = scaler.fit_transform(user_item_matrix)
```

### 4. Train-Test Split
Membagi data menjadi training (80%) dan testing (20%) untuk evaluasi model yang lebih akurat.

```python
train_data, test_data = train_test_split(ratings_clean, test_size=0.2, random_state=42)
```

### 5. TF-IDF Vectorization
Mengubah teks genre menjadi vektor numerik menggunakan TF-IDF untuk menghitung similarity antar film.

```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_clean['genres'])
```

## Modeling and Result

### Model 1: Content-Based Filtering

**Cara Kerja**:
Content-Based Filtering merekomendasikan item berdasarkan kesamaan atribut dengan item yang disukai pengguna sebelumnya. Pada implementasi ini, menggunakan fitur genre film yang diubah menjadi vektor TF-IDF, kemudian dihitung cosine similarity antar film.

**Parameter**:
- `stop_words='english'`: Mengabaikan kata umum dalam bahasa Inggris
- Default parameters untuk TF-IDF dan cosine similarity

**Hasil Rekomendasi untuk "Toy Story (1995)":**
1. Toy Story 2 (1999)
2. Adventures of Rocky and Bullwinkle, The (2000) 
3. Emperor's New Groove, The (2000)
4. Monsters, Inc. (2001)
5. Wild, The (2006)

### Model 2: Collaborative Filtering

**Cara Kerja**:
Collaborative Filtering memprediksi preferensi pengguna berdasarkan rating dari pengguna lain yang memiliki similarity. Menggunakan neural network dengan embedding layers untuk mempelajari representasi latent dari users dan items.

**Parameter**:
- `embedding_size=50`: Dimensi embedding vector
- `batch_size=64`: Ukuran batch untuk training
- `epochs=10`: Jumlah epoch training
- `optimizer='adam'`: Optimizer Adam dengan learning rate default
- `loss='mean_squared_error'`: Loss function untuk regression task

**Arsitektur Model**:
```
Input Layers → Embedding Layers → Flatten → Dot Product → Output
```

**Hasil Rekomendasi untuk User 1:**
1. Terminator 2: Judgment Day (1991)
2. Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
3. Godfather: Part II, The (1974)
4. Grand Day Out with Wallace and Gromit, A (1989)
5. Harold and Maude (1971)

### Kelebihan dan Kekurangan
**Content-Based Filtering**:
- ✅ Tidak memerlukan data dari pengguna lain
- ✅ Dapat merekomendasikan item baru
- ❌ Terbatas pada fitur yang tersedia
- ❌ Kurang mampu menemukan hubungan yang tidak terduga

**Collaborative Filtering**:
- ✅ Dapat menemukan hubungan yang tidak terduga
- ✅ Personalisasi yang lebih baik
- ❌ Cold start problem untuk user/item baru
- ❌ Memerlukan data yang cukup dari banyak pengguna

## Evaluation

### Metrik Evaluasi

**Content-Based Filtering**:
- **Precision**: TP / (TP + FP) = 0.85
- **Recall**: TP / (TP + FN) = 0.72

**Collaborative Filtering**:
- **MAE (Mean Absolute Error)**: (1/n) * Σ|y_i - ŷ_i| = 0.1539
- **RMSE (Root Mean Square Error)**: √[(1/n) * Σ(y_i - ŷ_i)²] = 0.1789
- **Precision@5**: 0.60
- **Precision@10**: 0.50

### Hasil Evaluasi

**Content-Based Filtering** menunjukkan precision yang tinggi (0.85) dalam merekomendasikan film dengan genre serupa. Hasil ini menunjukkan bahwa model efektif dalam menemukan film dengan karakteristik konten yang mirip.

**Collaborative Filtering** memiliki error yang rendah (MAE: 0.1539) dalam memprediksi rating dan dapat memberikan rekomendasi yang personal. Precision@5 sebesar 0.60 menunjukkan bahwa 60% dari 5 rekomendasi teratas relevan untuk pengguna.

### Dampak terhadap Business Understanding

**Problem Statement 1**: Information Overload
- ✅ Teratasi dengan menyediakan rekomendasi terfilter berdasarkan preferensi
- ✅ Pengguna tidak perlu menelusuri semua film yang tersedia

**Problem Statement 2**: Personalization Need  
- ✅ Terpenuhi melalui personalisasi berdasarkan historical rating (collaborative)
- ✅ Personalisasi berdasarkan konten yang disukai (content-based)

**Problem Statement 3**: Discovery Challenge
- ✅ Teratasi dengan merekomendasikan film baru yang sesuai selera
- ✅ Baik content-based maupun collaborative filtering berhasil memberikan discovery

**Goals**:
1. ✅ Sistem content-based filtering berhasil dibangun dengan precision 0.85
2. ✅ Sistem collaborative filtering berhasil dibangun dengan MAE 0.1539  
3. ✅ Memberikan rekomendasi relevan yang meningkatkan pengalaman pengguna

Kedua pendekatan berhasil mencapai tujuan yang ditetapkan dan memberikan dampak positif dalam menyelesaikan permasalahan bisnis yang diidentifikasi.

## Conclusion

Sistem rekomendasi yang dibangun telah berhasil memberikan rekomendasi film yang relevan menggunakan dua pendekatan yang berbeda. Content-based filtering efektif untuk merekomendasikan film dengan genre serupa, sementara collaborative filtering memberikan personalisasi yang lebih baik berdasarkan preferensi pengguna lain.

Untuk pengembangan selanjutnya, dapat dipertimbangkan untuk mengimplementasikan hybrid approach yang menggabungkan kedua metode tersebut, serta menambahkan fitur-fitur tambahan seperti director, actor, dan keywords untuk meningkatkan akurasi rekomendasi.

## References
1. Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. Springer.
2. MovieLens Dataset: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
3. TensorFlow Documentation: https://www.tensorflow.org/recommenders
4. Scikit-learn Documentation: https://scikit-learn.org/stable/

---Ini adalah bagian akhir laporan---
