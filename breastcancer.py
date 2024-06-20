# %% [markdown]
# UJIAN TENGAH SEMESTER
# 
# > **Klasifikasi Kanker Payudara Menggunakan Algoritma K-Nearest Neighbors (KNN) pada Dataset Breast-W**
# 
# DISUSUN OLEH
# 
# 1. Nama  : I Wayan Aditya Prayana Putra
# 2. NIM   : 21.12.1860

# %% [markdown]
# # Pra-Pemrosesan Data

# %% [markdown]
# 2. Membaca Data menggunakan library pandas

# %%
import pandas as pd

# Membaca dataset yang terdapat pada drive menggunakan pandas as pd
df = pd.read_csv('breastcancer.csv')
df.head()

# %%
# Menampilkan informasi ringkasan tentang DataFrame df, termasuk tipe data
df.info()

# %%
# Mengkonversi kolom 'Bare_Nuclei' dalam DataFrame df ke tipe data numerik
# Nilai yang tidak dapat dikonversi akan diubah menjadi NaN
df['Bare_Nuclei'] = pd.to_numeric(df.Bare_Nuclei, errors="coerce")

# Menampilkan informasi ringkasan tentang DataFrame df, termasuk tipe data
df.info()

# %% [markdown]
# 3. Pengecekan Missing Values

# %%
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Membaca file CSV berdasarkan data df yang sudah dipanggil diatas dan dimasukan ke variable baru
cardata = df

# Menampilkan DataFrame
print("DataFrame:")
print(cardata)

# Menampilkan Jumlah Missing Values per Kolom
print("Jumlah Missing Values per Kolom:")
print(cardata.isnull().sum())

# %%
# Menghapus semua baris dalam DataFrame df yang mengandung nilai NaN
# inplace=True berarti perubahan akan diterapkan langsung pada DataFrame df
df.dropna(inplace=True)

# %%
# Membaca file CSV berdasarkan data df yang sudah dipanggil diatas dan dimasukan ke variable baru
cardata = df

# Menampilkan DataFrame
print(cardata)

# Menampilkan Jumlah Missing Values per Kolom
print("Jumlah Missing Values per Kolom:")
print(cardata.isnull().sum())

# %% [markdown]
# 4. Pengecekan Duplikasi Data

# %%
# Mengecek duplikasi
duplicates = df.duplicated()

# Menampilkan hasil pengecekan
print("Baris yang merupakan duplikat:")
print(duplicates)

total_duplicates = duplicates.sum()
print("Total data duplikat dalam dataset :", total_duplicates)

# %% [markdown]
# 4. Imbalanced Class

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Mengecek distribusi kelas dengan value_counts()
class_distribution = df['Class'].value_counts()
print(class_distribution)

# Visualisasi distribusi kelas dengan Matplotlib
class_distribution.plot(kind='bar')
plt.title('Distribusi Kelas')
plt.xlabel('Kelas')
plt.ylabel('Jumlah')
plt.show()

# Visualisasi distribusi kelas dengan Seaborn
sns.countplot(x='Class', data=df)
plt.title('Distribusi Kelas')
plt.show()

# Menggunakan collections.Counter untuk menghitung distribusi kelas
class_distribution_counter = Counter(df['Class'])
print(class_distribution_counter)

# Menghitung rasio ketidakseimbangan
if len(class_distribution) > 1:
    imbalance_ratio = class_distribution[0] / class_distribution[1]
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
else:
    print("Hanya Ada satu kelas dalam dataset.")

# %%
# prompt: menangani imbalanced class dan masukan dia ke variable baru yaitu df

# Oversampling dengan SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

# Pisahkan fitur dan target
X = df.drop('Class', axis=1)
y = df['Class']

# Lakukan oversampling
X_resampled, y_resampled = smote.fit_resample(X, y)

# Gabungkan fitur dan target yang sudah dioversampling
df_oversampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)

# Tampilkan distribusi kelas setelah oversampling
class_distribution_oversampled = df_oversampled['Class'].value_counts()
print(class_distribution_oversampled)

# Visualisasikan distribusi kelas setelah oversampling
class_distribution_oversampled.plot(kind='bar')
plt.title('Distribusi Kelas setelah Oversampling')
plt.xlabel('Kelas')
plt.ylabel('Jumlah')
plt.show()

# Masukkan data yang sudah dioversampling ke dalam variable baru
df = df_oversampled

# %%
import pandas as pd

# Fungsi untuk mengubah teks menjadi angka 0 dan 1
def text_to_binary(text):
    if text.lower() == 'benign':
        return 0
    elif text.lower() == 'malignant':
        return 1
    else:
        raise ValueError("Input tidak valid, harus 'benign' atau 'malignant'.")

# Membuat DataFrame dari data
df = pd.DataFrame(df)

# Menghapus kolom 'id' dari DataFrame df
df.drop(columns='id', inplace=True)

# Menggunakan fungsi text_to_binary untuk mengubah kolom 'Class' menjadi angka
df['Class'] = df['Class'].apply(text_to_binary)

# Menampilkan lima baris pertama dari DataFrame
print("DataFrame setelah transformasi:")
print(df.head())


# %% [markdown]
# # Pemodelan

# %% [markdown]
# 1. Import Library

# %%
# Import Library yang dibutuhkan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced

# %% [markdown]
# 2. Menentukan Variabel Atribut dan Kelas

# %%
# Menentukan Variabel X (Fitur/Atribut) dan Variabel y (Kelas/Label)

X= df.drop(columns = 'Class').copy()
y= df['Class']

pd.DataFrame(y).head()

# %% [markdown]
# 3. Membagi Data untuk pengujian

# %%
# Membagi data menjadi data training dan data testing
# Data untuk testing 20%, data untuk training 80%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# %% [markdown]
# 5. Pembuatan Model

# %%
# Inisiasi Model
model = KNeighborsClassifier(n_neighbors=3)

# Training model dengan .fit()
model.fit(X_train, y_train)

# %% [markdown]
# 6. Melakukan Prediksi pada data Test

# %%
# Prediksi pada data test

y_pred = model.predict(X_test)
y_pred

# %%
y_test

# %% [markdown]
# 7. Melakukan Pengecekan Akurasi Prediksi

# %%
model_score = model.score(X_test, y_test)
print("KNN Test Score: {:.2f}%".format(model_score * 100))

# %%
# Memeriksa antara hasil prediksi dan data aktual

df = pd.DataFrame({'Prediksi': y_pred, 'Aktual': y_test})
df

# %%
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix

# %% [markdown]
# 8. Mencari Nilai K yang OPTIMAL

# %%
# Menentukan range dari jumlah tetangga yang akan diuji dalam model KNN
k_neighbors = range(1, 15)

# List untuk menyimpan akurasi pelatihan dan pengujian untuk setiap nilai k
train_accuracy = []
test_accuracy = []

# Melakukan iterasi untuk setiap nilai k dalam range yang telah ditentukan
for n in k_neighbors:
    # Membuat instance KNeighborsClassifier dengan jumlah tetangga saat ini
    m = KNeighborsClassifier(n_neighbors=n)

    # Melatih model menggunakan data pelatihan
    m.fit(X_train, y_train)

    # Menghitung dan menyimpan akurasi model pada data pelatihan
    train_accuracy.append(m.score(X_train, y_train))

    # Menghitung dan menyimpan akurasi model pada data pengujian
    test_accuracy.append(m.score(X_test, y_test))

# %%
# Memplot grafik akurasi pelatihan untuk setiap nilai k
plt.plot(k_neighbors, train_accuracy, label="Training Accuracy", marker='o', linestyle='-')

# Memplot grafik akurasi pengujian untuk setiap nilai k
plt.plot(k_neighbors, test_accuracy, label="Testing Accuracy", marker='o', linestyle='-')

# Menambahkan judul pada grafik
plt.title('KNN Model Accuracy')

# Menambahkan label pada sumbu x
plt.xlabel('Number of Neighbors (k)')

# Menambahkan label pada sumbu y
plt.ylabel('Accuracy')

# Menambahkan legenda untuk membedakan garis pelatihan dan pengujian
plt.legend()

# Menambahkan grid pada grafik untuk memudahkan pembacaan
plt.grid(True)

# Menampilkan grafik
plt.show()

# %%
# Menentukan akurasi pelatihan tertinggi dari list train_accuracy
max_train_accuracy = max(train_accuracy)

# Mencari indeks dari akurasi pelatihan tertinggi tersebut
max_train_index = train_accuracy.index(max_train_accuracy)

# Menentukan nilai k terbaik untuk data pelatihan berdasarkan indeks yang ditemukan
best_k_train = list(k_neighbors)[max_train_index]

# Menentukan akurasi pengujian tertinggi dari list test_accuracy
max_test_accuracy = max(test_accuracy)

# Mencari indeks dari akurasi pengujian tertinggi tersebut
max_test_index = test_accuracy.index(max_test_accuracy)

# Menentukan nilai k terbaik untuk data pengujian berdasarkan indeks yang ditemukan
best_k_test = list(k_neighbors)[max_test_index]

# Mencetak hasil akurasi tertinggi pada data pelatihan dan nilai k yang sesuai
print("KNN Train Accuracy: {:.2f}% with k = {}".format(max_train_accuracy * 100, best_k_train))

# Mencetak hasil akurasi tertinggi pada data pengujian dan nilai k yang sesuai
print("KNN Test Accuracy: {:.2f}% with k = {}".format(max_test_accuracy * 100, best_k_test))

# %% [markdown]
# 9. EVALUASI MODEL

# %%
from sklearn.metrics import accuracy_score, classification_report
from imblearn.metrics import classification_report_imbalanced


# Evaluasi Kinerja
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Classification Report Imbalanced:")
print(classification_report_imbalanced(y_test, y_pred))

# %%
# prompt: prediksi model input

input_data = (2,5,4,8,3,3,3,1,4)

# reshape the input data
input_data_reshaped = np.reshape(input_data, (1, -1))

# standardize the input data
std_data = scaler.transform(input_data_reshaped)

# predict the class
prediction = model.predict(std_data)

# print the prediction
print(prediction)

if (prediction[0] == 0):
  print('The tumor is Benign ')
else:
  print('The tumor is Malignant')


# %%
# prompt: save model

import pickle

# Save the model to a file
with open('breastcancernew.sav', 'wb') as file:
    pickle.dump(model, file)



