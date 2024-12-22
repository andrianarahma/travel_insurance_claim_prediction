# travel_insurance_claim_prediction

**Latar Belakang**
Asuransi perjalanan adalah salah satu asuransi umum yang ditawarkan untuk memberikan perlindungan jika terjadi sesuatu yang tidak diinginkan selama perjalanan, baik perjalanan domestik maupun internasional. Besaran premi yang dibayar bergantung pada lamanya perjalanan, plan yang diinginkan, dan tujuan perjalanan. Beberapa negara mewajibkan orang-orang yang ingin melakukan keberangkatan untuk memiliki asuransi perjalanan, seperti di Eropa dan Amerika.

**Rumusan Masalah**
Lambatnya proses pencairan klaim asuransi membuat pelanggan tidak merasa puas atas pelayanan dari perusahaan asuransi. Hal ini berpotensi membuat pelanggan tersebut tidak melanjutkan membeli polis asuransi.

**Tujuan Analisis dan Model ML**
Untuk memprediksi apakah seseorang akan mengajukan klaim asuransi atau tidak. Diharapkan dengan adanya prediksi dengan model ML, maka perusahaan asuransi dapat mengetahui lebih dulu sehingga dapat menyediakan dana yang cukup untuk proses pencairan klaim asuransi. Dengan demikian, proses pencairan klaim asuransi dapat berjalan lebih cepat. Misalkan biasanya dicairkan dalam 10 hari kerja, maka dengan adanya prediksi ini, maka pengajuan klaim dapat dicairkan kurang dari 10 hari kerja. Selain itu, dengan adanya prediksi ini, maka perusahaan asuransi dapat menentukan besar premi dan ketentuan polis asuransi untuk masa yang akan datang.

**Data yang Dimiliki**
1. Agency : nama agensi
2. Agency Type : tipe agensi
3. Distribution Channel : channel yang disediakan oleh agensi
4. Product Name : nama produk asuransi perjalanan
5. Gender : jenis kelamin yang ditanggung
6. Duration : lama waktu perjalanan
7. Destination : negara tujuan perjalanan
8. Net Sales : jumlah penjualan polis asuransi
9. Commission (in value) : komisi yang diterima agensi
10. Age : usia yang ditanggung
11. Claim : status klaim

**Target dan Metrik yang Digunakan**

**Menentukan Target**
Karena kita akan memprediksi apakah pemilik polis akan melakukan klaim atau tidak, maka kolom yang menjadi target adalah kolom 'Claim' dengan :
- positif (Y = 1) adalah klaim
- negatif (Y = 0) adalah tidak klaim

**Hubungan antara hasil prediksi dengan data aktual :**
- FP (False Positive) : hasil prediksi pemilik polis akan klaim, padahal tidak klaim
- FN (False Negative) : hasil prediksi pemilik polis tidak klaim, padahal klaim
- TP (True Positive) : hasil prediksi dan aktual adalah pemilik polis akan klaim
- TF (True Negative) : hasil prediksi dan aktual adalah pemilik polis tidak akan klaim

**Menentukan Metrik yang Digunakan**
- Jika yang terjadi adalah FP, maka risiko yang akan diterima oleh perusahaan asuransi adalah :
    - menyediakan dana untuk persiapan klaim dari pemilik polis, sehingga premi yang ada tidak dapat diinvestasikan. 

- Jika yang terjadi adalah FN, maka risiko yang akan diterima oleh perusahaan asuransi adalah :
    - pelanggan berhenti membeli polis karena merasa tidak puas atas pelayanan perusahaan asuransi akibat dari lamanya proses pencairan klaim asuransi.
    - berhentinya pelanggan mengakibatkan hilangnya dana yang dapat diinvestasikan oleh pihak perusahaan.
    - diperlukan budget tambahan untuk marketing (mencari pelanggan baru)

Dari penjabaran risiko di atas, risiko yang lebih besar akan diterima oleh perusahaan asuransi jika terjadi FN. Oleh karena itu, kita akan menggunakan metrik untuk mereduksi terjadinya FN, yaitu **recall**.

**Temuan pada Data**
- Terdapat lebih dari 70% missing values pada kolom 'Gender', sehingga kolom 'Gender' akan didrop.
- Terdapat data anomali, yaitu terdapat nilai negatif pada kolom 'Duration' dan 'Net Sales', dimana Duration dan Net Sales tidak mungkin bernilai negatif, sehingga data tersebut akan didrop.
- Terdapat data duplikat, sehingga data tersebut juga akan didrop.
- Terdapat outliers ekstrem pada kolom 'Duration', 'Net Sales', dan 'Age', sehingga outliers tersebut akan didrop.
- Terdapat Duration yang lebih dari 1 tahun atau 365 hari. Tapi, untuk kasus ini data tersebut akan dibiarkan.
- Jumlah data target tidak seimbang, dimana data Yes (yang mengajukan klaim asuransi) ada sebanyak 1,73%, sedangkan No (yang tidak mengajukan klaim asuransi) ada sebanyak 98,27%, sehingga akan dilakukan balancing.

**Menentukan Kolom Features**
1. Kolom kategorik dicek pengaruhnya terhadap kolom target dengan menggunakan chi-square. Diperoleh kolom kategorik yang dijadikan feature adalah:
   - 'Agency'
   - 'Agency Type'
   - 'Product Name'
   - 'Destination'
2. Kolom numerik dicek korelasinya terhadap kolom target dengan menggunakan uji korelasi dengan sebelumnya kolom target di-encoding menjadi 1 dan 0. Y = 1 untuk Yes dan Y = 0 untuk No. Diperoleh kolom numerik yang dijadikan feature adalah:
   - 'Duration'
   - 'Net Sales'
   - 'Commision (in value)
  
**Feature Engineering**
1. Kolom kategorik yang dijadikan feature akan di-encoding. Jika banyak unik (n-unique) dari kolom tersebut <= 5, maka digunakan metode One Hot Encoder. Jika n-unique > 5, maka digunakan metode Binary Encoder. Keduanya digunakan untuk kolom yang tidak mempunyai urutan. Pada data ini, tidak ada kolom kategorik yang mempunyai urutan.
2. Kolom numerik yang dijadikan feature akan di-scaling menggunakan robust scaler. Karena metode robust scaler lebih tahan terhadap outliers.

**Model yang Digunakan**
Karena keterbatasan waktu pengerjaan, model yang digunakan untuk dibandingkan adalah :
- DecisionTree
- KNN
- Logistic Regression
- Naive Beiyes
- Hard Vote
- Stacking
- Random Forest
- Adaboost
- Bagging
- XG Boost
- Gradient Boost

**Metode Balancing yang Digunakan**
Karena keterbatasan waktu pengerjaan, metode balancing yang digunakan adalah:
- SMOTE
- ADASYN
- Random Over Sampler
- Random Under Sampler
- Near Miss
- Tome k-Links
- Borderline SMOTE

**Recall Score yang Diperoleh**
Dengan membandingkan semua model dan metode balancing, diperoleh Top 5 Recall Score:
![image](https://github.com/user-attachments/assets/7b12a346-aa02-4861-831a-048d404340bb)

**Hyperparameter Tuning**
Untuk Top 5 Recall Score tersebut, dilakukan hyperparameter tuning untuk masing-masing model. Di sini, digunakan metode Randomized Search karena wkatu pengerjaan yang terbatas. Untuk lebih optimal, bisa digunakan metode Grid Search.
1. Hyperparameter Tuning untuk Model KNN dengan Random Under Sampler
   Best parameter: p = 1; n-neighbors = 9; leaf size = 36
   Best recall score: 0.758
2. Hyperparameter Tuning untuk Model Stacking dengan Near Miss
   Best parameter: final_estimator: DecisionTreeClassifier(random_state=2024), cv = 5
   Best recall score: 0.733
3. Hyperparameter Tuning untuk Model Naive Beiyes dengan ADASYN
   Best parameter: var_smoothing: np.float64(1.519911082952933e-09)
   Best recall score: 0.726
4. Hyperparameter Tuning untuk Model Naive Beiyes dengan Random Under Sampler
   Best parameter: var_smoothing: np.float64(2.310129700083158e-09)
   Best recall score: 0.681
5. Hyperparameter Tuning untuk Model Gradient Boost dengan Random Under Sampler
   Best parameter: n_estimators: 35; max depth: 5; learning rate: 0.15
   Best recall score: 0.735

Dari hasil recall score setelah hyperparameter tuning, dipilih recall score tertinggi, yaitu model KNN dengan Random Under Sampler, dengan parameter p = 1; n-neighbors = 9; leaf size = 36.

**Aplikasikan ke Data Tes**
Model yang telah dipilih dari hasil hyperparameter tuning, diaplikasikan ke data tes untuk melihat performanya pada data tes.
Diperoleh recall score-nya adalah 0,759.

**Kesimpulan**
1. Data target sangat tidak seimbang
2. Setelah dilakukan perbandingan model machine learning, diperoleh recall score terbaik adalah dengan menggunakan model KNN dengan metode resampling random under sampler.
3. Model yang digunakan masih belum bekerja maksimal, walaupun recall score yang diperoleh sudah melebihi 70%.

**Saran dan Rekomendasi**
1. Lebih baik ditambahkan data lagi untuk mengurangi ketimpangan jumlah data target.
2. Lebih baik ditambahkan feature lainnya, seperti besaran premi yang dibayarkan, tingkat risiko perjalanan, kondisi kesehatan (riwayat penyakit kronis, dsb), pendapatan pelanggan, dll.
3. Dapat ditambahkan model lainnya saat perbandingan jika waktu pengerjaan lebih lama karena model yang digunakan ini terbatas disebabkan keterbatasan waktu pengerjaan.
4. Ketika melakukan tuning, dapat digunakan metode grid search agar model lebih optimal jika waktu pengerjaan lebih lama. Pada pengerjaan ini, digunakan metode randomized untuk tuning dikarenakan keterbatasan waktu.

Sampai di sini dulu penjelasan tentang pemodelan machine learning untuk memprediksi apakah seseorang akan klaim asuransi perjalanannya atau tidak. Model yang digunakan di sini masih jauh dari kata sempurna, tapi semoga bermanfaat.

Thank you and have a nice day!
