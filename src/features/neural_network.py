from src.dataset import dataset
import pandas as pd

# Pour lancer le code, se mettre dans le dossier defensive et faire la commande "python3 -m models.neural_network"

data = dataset.get_dataset()
# Data est un dictionnaire avec les clefs "train" et "test". Train est un pandas, test aussi.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger les données
data = dataset.get_dataset()
data_train = data["train"]

# Séparer les caractéristiques et les labels
text_columns = ["subject", "body", "adress", "domain", "domain_extension"]
numeric_columns = [
    col for col in data_train.columns if col not in text_columns + ["ground_truth"]
]

X_numeric = data_train[numeric_columns].values
y = data_train["ground_truth"]

# Prétraitement du texte
MAX_WORDS = 5000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
texts = data_train[text_columns].astype(str).apply(lambda x: " ".join(x), axis=1)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X_text = pad_sequences(sequences, maxlen=MAX_LEN)

# Normalisation des données numériques
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X_numeric)

# Fusion des données textuelles et numériques
X = np.hstack((X_numeric, X_text))

# Division en train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Construction du réseau de neurones
model = keras.Sequential(
    [
        layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compilation du modèle
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entraînement du modèle
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))

# Évaluation du modèle
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Précision du modèle : {accuracy:.4f}")

# Charger les données de test
data_test = data["test"]

# Séparer les caractéristiques et labels
X_test_numeric = data_test[numeric_columns].values
y_test = data_test["ground_truth"]

# Prétraitement du texte
texts_test = data_test[text_columns].astype(str).apply(lambda x: " ".join(x), axis=1)
sequences_test = tokenizer.texts_to_sequences(texts_test)
X_test_text = pad_sequences(sequences_test, maxlen=MAX_LEN)

# Normaliser les données numériques
X_test_numeric = scaler.transform(X_test_numeric)

# Fusionner les données textuelles et numériques
X_test = np.hstack((X_test_numeric, X_test_text))

# Évaluer le modèle sur les données de test
loss_test, accuracy_test = model.evaluate(X_test, y_test)
print(f"Précision du modèle sur le test : {accuracy_test:.4f}")

# Précision du modèle : 0.6823
# Précision du modèle sur le test : 0.4815
