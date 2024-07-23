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

## Penjelasan Dataset
Dataset = weatherAUS.csv
Dataset yang digunakan berisi data cuaca harian dari berbagai lokasi di Australia
Fitur Numerik:

    MinTemp: Suhu minimum pada hari itu di derajat Celsius.
    MaxTemp: Suhu maksimum pada hari itu di derajat Celsius.
    Rainfall: Jumlah curah hujan yang tercatat pada hari itu dalam mm.
    Evaporation: Tingkat penguapan dari tanah dan penguapan dari vegetasi terbuka, biasanya dalam mm.
    Sunshine: Jumlah jam sinar matahari yang bersinar pada hari itu.
    WindGustSpeed: Kecepatan angin tertinggi pada hari itu dalam km/jam.
    WindSpeed9am: Kecepatan angin pada pukul 9 pagi dalam km/jam.
    WindSpeed3pm: Kecepatan angin pada pukul 3 sore dalam km/jam.
    Humidity9am: Kelembapan relatif pada pukul 9 pagi (persen).
    Humidity3pm: Kelembapan relatif pada pukul 3 sore (persen).
    Pressure9am: Tekanan atmosfer pada pukul 9 pagi dalam hectopascals.
    Pressure3pm: Tekanan atmosfer pada pukul 3 sore dalam hectopascals.
    Cloud9am: Persentase langit yang tertutup oleh awan pada pukul 9 pagi.
    Cloud3pm: Persentase langit yang tertutup oleh awan pada pukul 3 sore.
    Temp9am: Suhu pada pukul 9 pagi di derajat Celsius.
    Temp3pm: Suhu pada pukul 3 sore di derajat Celsius.

Fitur Kategorikal:

    Location: Lokasi pengamatan cuaca.
    WindGustDir: Arah angin terkencang selama hari itu.
    WindDir9am: Arah angin pada pukul 9 pagi.
    WindDir3pm: Arah angin pada pukul 3 sore.
    RainToday: Indikator apakah hujan turun atau tidak pada hari itu ("Yes" atau "No").

# Exploratory Data Analysis (EDA)
