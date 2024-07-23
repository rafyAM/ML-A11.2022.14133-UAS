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

    - MinTemp: Suhu minimum pada hari itu di derajat Celsius.
    
    - MaxTemp: Suhu maksimum pada hari itu di derajat Celsius.
    
    - Rainfall: Jumlah curah hujan yang tercatat pada hari itu dalam mm.
    
    - Evaporation: Tingkat penguapan dari tanah dan penguapan dari vegetasi terbuka, biasanya dalam mm.
    
    - Sunshine: Jumlah jam sinar matahari yang bersinar pada hari itu.
    
    - WindGustSpeed: Kecepatan angin tertinggi pada hari itu dalam km/jam.
    
    - WindSpeed9am: Kecepatan angin pada pukul 9 pagi dalam km/jam.
    
    - WindSpeed3pm: Kecepatan angin pada pukul 3 sore dalam km/jam.
    
    - Humidity9am: Kelembapan relatif pada pukul 9 pagi (persen).
    
    - Humidity3pm: Kelembapan relatif pada pukul 3 sore (persen).
    
    - Pressure9am: Tekanan atmosfer pada pukul 9 pagi dalam hectopascals.
    
    - Pressure3pm: Tekanan atmosfer pada pukul 3 sore dalam hectopascals.
    
    - Cloud9am: Persentase langit yang tertutup oleh awan pada pukul 9 pagi.
    
    - Cloud3pm: Persentase langit yang tertutup oleh awan pada pukul 3 sore.
    
    - Temp9am: Suhu pada pukul 9 pagi di derajat Celsius.
    
    -Temp3pm: Suhu pada pukul 3 sore di derajat Celsius.

Fitur Kategorikal:

    - Location: Lokasi pengamatan cuaca.
    
    - WindGustDir: Arah angin terkencang selama hari itu.
    
    - WindDir9am: Arah angin pada pukul 9 pagi.
    
    - WindDir3pm: Arah angin pada pukul 3 sore.
    
    - RainToday: Indikator apakah hujan turun atau tidak pada hari itu ("Yes" atau "No").

# Exploratory Data Analysis (EDA)

## Memuat data
```python
data = pd.read_csv("weatherAUS.csv")
```
## Pemeriksaan awal
```python
print(data.head())
print(data.describe())
```
## Mengatasi nilai yang hilang
```python
data.fillna(data.median(), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)
```
## Menghapus duplikat
```python
data.drop_duplicates(inplace=True)
```
## Visualisasi distribusi dan hubungan
```python
plt.figure(figsize=(10, 6))
sns.histplot(data['Rainfall'], kde=True)
sns.pairplot(data[['MaxTemp', 'Rainfall', 'Humidity9am']])
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
```

## Encoding dan Skala
```python
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
data_processed = preprocessor.fit_transform(data)
```

## Pembagian data
```python
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
# Proses Features Dataset
- feature Selection: Memilih fitur yang relevan untuk model termasuk Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow. Fitur tambahan seperti Day, Month, dan year
- Feature Engineering: Ekstraksi informasi tambahan dari fitur yang ada untuk meningkatkan kualitas prediksi.

# Proses Learning / Modeling

1. Persiapan Data
   - memisahakn fitur X dan target Y
     ```python
     features = data.drop('RainTomorrow', axis=1)
     labels = data['RainTomorrow']
     ```
   - Membagi data menjadi set train dan test
     ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)
     ```
2. Preprocessing Pipeline
   - Membuat pipeline untuk fitur numerik dan kategorikal
     ```pyhton
     num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean')),('scale', StandardScaler())])

     cat_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')),('encoder', OrdinalEncoder())])
     ```
   - Menggabungkan pipline menggunakn ColumnTransformer
     ```python
     col_transformer = ColumnTransformer(
     transformers=[
        ('num_pipeline', num_pipeline, num_col),
        ('cat_pipeline', cat_pipeline, cat_col)
     ],
     remainder='passthrough',
     n_jobs=-1
     )
     ```
3. Model Utama Random Forest Classifier
   - Inisialisai model
     ```python
     rf = RandomModelClassifier( random_state=42)
     ```

   - Membuat pipeline final yang menggabungkan preprocessing dan model
     ```python
     rf = RandomForestClassifier(random_state=42)
     ```
4. Pelatihan Model
   - Melatih model menggunakan data pelatihan
     ```python
     pipefinal.fit(X_train, y_train)
     ```
5. Prediksi
   - Membuat prediksi menggunakan data pengujian
     ```python
     pred = pipefinal.predict(X_test)
     ```
6. Evalusi Model
   - Menghitung matrix evaluasi
     ```python
     from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

     print('Accuracy Score:', accuracy_score(y_test, pred))
     print('Classification Report:\n', classification_report(y_test, pred))
     ```
   Accuracy Score : 0.8624437781109445 

Classification Report : 
               precision    recall  f1-score   support

          No       0.88      0.96      0.92     18909
         Yes       0.77      0.51      0.61      5103

    accuracy                           0.86     24012
   macro avg       0.82      0.73      0.76     24012
weighted avg       0.85      0.86      0.85     24012
     
   - Melakukan Cross validation
     ```python
     from sklearn.model_selection import cross_val_predict

     y_pred = cross_val_predict(pipefinal, X_train, y_train, cv=3)
     cm = confusion_matrix(y_train, y_pred)
     print('Confusion Matrix:\n', cm)
     ```
    array([[42116,  1890],
       [ 5987,  6035]])
     
7. Visualisasi ROC Curve
   ```python
   from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    y_prob = pipefinal.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='Yes')
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
   ```
   ![feature importance](https://github.com/rafyAM/ML-A11.202214133-UAS/blob/main/images/ReceiverOperatingCharacteristic(ROC).png?raw=true)

8. Model tambahan untuk klasifikasi awan, suhu, dan kabut
   - Membuat fungsi klasifikasi
     ```python
     def classify_cloud(row):
        return 'Overcast' if row['Cloud9am'] >= 7 or row['Cloud3pm'] >= 7 else 'Cloudy' if row['Cloud9am'] >= 4 or row['Cloud3pm'] >= 4      else 'Sunny'
     
     def classify_temperature(row):
            avg_temp = (row['MinTemp'] + row['MaxTemp']) / 2
            return 'Hot' if avg_temp >= 30 else 'Warm' if avg_temp >= 20 else 'Cool' if avg_temp >= 10 else 'Cold'
        
     def classify_fog(row):
            return 'Foggy' if row['Humidity9am'] >= 90 or row['Humidity3pm'] >= 90 else 'Not Foggy'
     ```
   - Menerapkan klasifikasi ke dataset
     ```python
     data['CloudClassification'] = data.apply(classify_cloud, axis=1)
     data['TempClassification'] = data.apply(classify_temperature, axis=1)
     data['FogClassification'] = data.apply(classify_fog, axis=1)
     ```
   - Melatih model untuk setiap klasifikasi
     ```pyhton
     features = ['Cloud9am', 'Cloud3pm', 'MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm']
     X = data[features]

     for classification in ['CloudClassification', 'TempClassification', 'FogClassification']:
     y = data[classification]
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
     f_model = RandomForestClassifier(random_state=42)
     rf_model.fit(X_train, y_train)
        
     y_pred = rf_model.predict(X_test)
        
     print(f"\n{classification} Evaluation:")
     print("Accuracy:", accuracy_score(y_test, y_pred))
     print("Classification Report:\n", classification_report(y_test, y_pred))
     ```
       

# Performa Model
1. Prediksi Prediksi Hujan Esok Hari dengan akurasi = 0.8624(86.24%):
   ```python
   pred=pipefinal.predict(x_test)
   print('Accuracy Score :', accuracy_score(y_test, pred) , '\n')
   print('Classification Report :', '\n',classification_report(y_test, pred))

   ```
       Accuracy Score : 0.8624437781109445 
    
    Classification Report : 
                   precision    recall  f1-score   support
    
              No       0.88      0.96      0.92     18909
             Yes       0.77      0.51      0.61      5103
    
        accuracy                           0.86     24012
       macro avg       0.82      0.73      0.76     24012
    weighted avg       0.85      0.86      0.85     24012
- Model memiliki akurasi keseluruhan yang baik (86.24%).
- Performa lebih baik dalam memprediksi "Tidak Hujan" (No) dibandingkan "Hujan" (Yes).
- Recall untuk kelas "Yes" relatif rendah (0.51), menunjukkan model kurang baik dalam mendeteksi kejadian hujan yang sebenarnya.
  
- Receiver Operating Characteristic (ROC)
  ![feature importance](https://github.com/rafyAM/ML-A11.202214133-UAS/blob/main/images/ReceiverOperatingCharacteristic(ROC).png?raw=true)


2.  Model Klasifikasi Awan:
   ```
    Akurasi: 1.0 (100%)
    Presisi: 1.0
    Recall: 1.0
    F1 Score: 1.0
    Interpretasi:
   ```
Model ini menunjukkan performa sempurna, yang mungkin mengindikasikan overfitting atau fitur yang terlalu prediktif.


4. Model Klasifikasi Suhu:
   ```
    Akurasi: 0.9952 (99.52%)
    Presisi: 0.9952
    Recall: 0.9952
    F1 Score: 0.9952
    Interpretasi:
   ```
    Model ini juga menunjukkan performa yang sangat tinggi, hampir sempurna.


6. Model Klasifikasi Kabut:
   ```
    Akurasi: 1.0 (100%)
    Presisi: 1.0
    Recall: 1.0
    F1 Score: 1.0
    Interpretasi:
   ```
    Seperti model klasifikasi awan, model ini juga menunjukkan performa sempurna.

![feature importance](https://github.com/rafyAM/ML-A11.202214133-UAS/blob/main/images/ConfusionMatrixKlasifikasiCuaca.png?raw=true)

# Diskusi Hasil dan Kesimpulan

## hasil :
Analisis dan pemodelan cuaca menggunakan Random Forest menghasilkan model dengan akurasi 86.24% dalam memprediksi kemungkinan hujan esok hari. Preprocessing data melibatkan penanganan nilai yang hilang dan outlier, serta pembuatan fitur tambahan seperti klasifikasi awan, suhu, dan kabut. Model menunjukkan performa lebih baik dalam memprediksi hari tanpa hujan dibandingkan hari hujan, yang mungkin disebabkan oleh ketidakseimbangan kelas dalam dataset. Kurva ROC menunjukkan performa klasifikasi yang baik dengan AUC tinggi. Klasifikasi tambahan menunjukkan akurasi sangat tinggi, namun perlu diwaspadai kemungkinan overfitting. Pengujian model dengan data baru menunjukkan kemampuan mempertimbangkan variasi geografis dan kondisi cuaca.

## Kesimpulan :
Model Random Forest yang dikembangkan menunjukkan performa yang baik dalam memprediksi cuaca, namun masih ada ruang untuk peningkatan, terutama dalam menyeimbangkan prediksi antara kelas hujan dan tidak hujan. Model ini dapat menjadi alat berguna untuk prediksi cuaca jangka pendek, tetapi sebaiknya digunakan bersama metode prediksi cuaca lainnya untuk hasil yang lebih akurat dan komprehensif. Pengembangan lebih lanjut dapat melibatkan feature engineering, teknik balancing dataset, validasi silang yang lebih ekstensif, dan integrasi data time series untuk memperhitungkan pola cuaca jangka panjang. Meskipun model menunjukkan akurasi tinggi, perlu kehati-hatian terhadap kemungkinan overfitting, terutama pada klasifikasi tambahan yang menunjukkan akurasi mendekati sempurna.
