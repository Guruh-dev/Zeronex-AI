# -- coding: utf-8 --
"""
Pipeline Pembangunan AI: Machine Learning (ML) dan Deep Learning (LLM)

Bagian 1: Machine Learning
  - Menggunakan dataset Iris untuk klasifikasi dengan RandomForestClassifier.
  - Evaluasi model dengan classification report, confusion matrix.
  - Visualisasi confusion matrix dan feature importances.

Bagian 2: Deep Learning (LLM)
  - Menggunakan pipeline text-generation dari HuggingFace (model GPT-2) untuk menghasilkan teks.
  - Analisis output teks (frekuensi kata) dan visualisasikan top 10 kata.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###############################################
# Bagian 1: Machine Learning dengan Dataset Iris
###############################################

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Memuat dataset Iris
iris = load_iris()
X = iris.data      # Fitur: sepal length, sepal width, petal length, petal width
y = iris.target    # Label: jenis bunga

# Memisahkan data menjadi training dan testing (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun dan melatih model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Melakukan prediksi pada data testing
y_pred = rf_model.predict(X_test)

# Evaluasi model: Classification Report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Classification Report:\n", report)

# Evaluasi model: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix - Iris Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Visualisasi Feature Importances
importances = rf_model.feature_importances_
features = iris.feature_names
plt.figure(figsize=(8, 4))
plt.bar(features, importances, color='green')
plt.title("Feature Importances pada Iris Dataset")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###############################################
# Bagian 2: Deep Learning (LLM) dengan GPT-2 untuk Text Generation
###############################################

from transformers import pipeline
import re
from collections import Counter

# Inisialisasi pipeline untuk text generation dengan model GPT-2
text_generator = pipeline('text-generation', model='gpt2')

# Prompt awal untuk menghasilkan teks
prompt = "Di masa depan, teknologi AI akan"
generated_output = text_generator(prompt, max_length=100, num_return_sequences=1)

# Ekstrak teks yang dihasilkan
generated_text = generated_output[0]['generated_text']
print("\nHasil Text Generation:\n", generated_text)

# Analisis sederhana: hitung frekuensi kata dari output teks
# Bersihkan teks dari tanda baca dan ubah ke huruf kecil
words = re.findall(r'\w+', generated_text.lower())
word_counts = Counter(words)
top_words = word_counts.most_common(10)
print("\nTop 10 kata beserta frekuensinya:", top_words)

# Visualisasi frekuensi kata (Top 10)
words_top, counts_top = zip(*top_words)
plt.figure(figsize=(10, 5))
plt.bar(words_top, counts_top, color='orange')
plt.title("Top 10 Frekuensi Kata pada Generated Text")
plt.xlabel("Kata")
plt.ylabel("Frekuensi")
plt.tight_layout()
plt.show()