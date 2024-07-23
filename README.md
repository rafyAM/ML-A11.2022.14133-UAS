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

## Memuat data
``python
data = pd.read_csv("weatherAUS.csv")
``
## Pemeriksaan awal
``python
print(data.head())
print(data.describe())
``
## Mengatasi nilai yang hilang
``python
data.fillna(data.median(), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)
``
## Menghapus duplikat
``python
data.drop_duplicates(inplace=True)
``
## Visualisasi distribusi dan hubungan
``python
plt.figure(figsize=(10, 6))
sns.histplot(data['Rainfall'], kde=True)
sns.pairplot(data[['MaxTemp', 'Rainfall', 'Humidity9am']])
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
``

## Encoding dan Skala
``python
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
data_processed = preprocessor.fit_transform(data)
``

## Pembagian data
``python
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
``
# Proses Features Dataset
- feature Selection: Memilih fitur yang relevan untuk model termasuk Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow. Fitur tambahan seperti Day, Month, dan year
- Feature Engineering: Ekstraksi informasi tambahan dari fitur yang ada untuk meningkatkan kualitas prediksi.

# Proses Learning / Modeling
1. Persiapan Data:
- Data dimuat dari file CSV: data = pd.read_csv("weatherAUS.csv")
- Nilai yang hilang ditangani menggunakan fungsi kustom impute_missing()
- Outlier ditangani menggunakan metode IQR dengan handle_outlires_IQR()
- Fitur tambahan dibuat: 'Day', 'Month', 'Year' dari kolom 'Date'
2. Rekayasa Fitur:
- Klasifikasi baru ditambahkan: 'CloudClassification', 'TempClassification', 'FogClassification'
3. Pembuatan Pipeline:
- Pipeline terpisah dibuat untuk fitur numerik dan kategorikal:
  ``python
  num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean')), ('scale', StandardScaler())])
  cat_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder())])
  ``
4. Pembagian Data
- Data dibagi menjadi fitur (X) dan variabel target (y):
  ``python
features = data.drop('RainTomorrow', axis=1)
labels = data['RainTomorrow']
  ``
- Pembagian data latih uji dilakukan:
  ``python
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)
  ``
5. Column Transformer:
- ColumnTransformer dibuat untuk menerapkan preprocessing yang sesuai pada kolom numerik dan kategorikal:
  ``python
col_transformer = ColumnTransformer(
    transformers=[
        ('num_pipeline', num_pipeline, num_col),
        ('cat_pipeline', cat_pipeline, cat_col)
    ],
    remainder='passthrough',
    n_jobs=-1
)
  ``
6. Pemilihan Model:
- Random Forest Classifier dipilih sebagai model utama:
  ``python
rf = RandomForestClassifier(random_state=42)
  ``
7. Pipeline Akhir:
- Column transformer dan model random forest digabungkan menjadi pipeline akhir:
  ``python
pipefinal = make_pipeline(col_transformer, rf)
  ``
8. Pelatihan model
- Pipeline akhir dilatih dengan data latih:
    ``
pipefinal.fit(x_train, y_train)
    ``
9. Evaluasi Model
- Prediksi dilakukan pada set uji:
  ``python
pred = pipefinal.predict(x_test)
  ``
- Berbagai metrik dihitung: Skor Akurasi, Laporan Klasifikasi (Presisi, Recall, F1-score), Matriks Konfusi, Kurva ROC dan AUC diplot
  
10. Model Tambahan
    - Model Random Forest terpisah dilatih untuk klasifikasi Awan, Suhu, dan Kabut
      
12. Pengujian Model
    - Model yang telah dilatih digunakan untuk membuat prediksi pada data baru yang belum pernah dilihat




