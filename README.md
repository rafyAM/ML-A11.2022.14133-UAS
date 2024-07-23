# Prediksi Cuaca di Australia Menggunakan Machine Learning
## Rafy Attala Mohamad - A11.2022.14133

## Ringkasan
Proyek ini berfokus pada memprediksi apakah akan terjadi hujan di berbagai lokasi di Australia pada hari berikutnya berdasarkan data historis cuaca. Data yang digunakan mencakup berbagai parameter meteorologi yang diukur setiap hari.

## Permasalahan
- Dataset mengandung banyak nilai yang hilang dan beberapa outlier yang dapat mempengaruhi keakuratan prediksi.
- Variabilitas cuaca yang tinggi dan keunikan kondisi lokal mempersulit model untuk membuat prediksi yang akurat secara konsisten.

## Tujuan yang akan dicapai
- Memprediksi kemungkinan hujan pada hari berikutnya dengan akurasi yang tinggi untuk membantu dalam perencanaan aktivitas dan manajemen sumber daya.
- Mengklasifikasikan kondisi cuaca seperti keberadaan awan, suhu, dan kabut untuk memberikan informasi lebih lanjut yang bisa berguna dalam berbagai aplikasi seperti pertanian dan lain lain

## Model/Alur Peneyelesaian
- Pengolahan Data Awal: Pembersihan dan penyiapan data melalui pengisian nilai yang hilang dan penghapusan outlier.
- Pengembangan Fitur: Mengubah tanggal ke dalam format numerik dan mengategorikan variabel berdasarkan parameter cuaca.
- Pembangunan dan Pelatihan Model: Menggunakan RandomForestClassifier karena robust terhadap overfitting dan efektif dalam mengelola fitur kategorikal dan numerik.
- Validasi Model: Penerapan cross-validation untuk menilai kestabilan dan reliabilitas model.
- Testing: Melakukan pengujian model dengan data yang belum pernah dilihat sebelumnya untuk mengevaluasi performa nyata model dalam kondisi operasional.
- Evaluasi Model: Penggunaan metrik seperti akurasi, precision, recall, F1-score, dan ROC-AUC untuk evaluasi komprehensif.
