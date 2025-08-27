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

### Content-Based Filtering
Menggunakan TF-IDF untuk mengubah genre menjadi vektor dan cosine similarity untuk menghitung kesamaan antar film.

**Hasil Rekomendasi untuk "Toy Story (1995)":**
1. Toy Story 2 (1999)
2. Adventures of Rocky and Bullwinkle, The (2000)
3. Emperor's New Groove, The (2000)
4. Monsters, Inc. (2001)
5. Wild, The (2006)

### Collaborative Filtering
Membangun neural network dengan embedding layers untuk mempelajari preferensi pengguna.

**Arsitektur Model:**
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
