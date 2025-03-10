import argparse
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import urllib.request
import zipfile
import os

# Expressions régulières et mots-clés spam
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
SPAM_KEYWORDS = {"free", "win", "money", "prize", "buy now", "urgent", "offer", "gift", "promotion"}

def download_dataset():
    """ Télécharge et extrait le dataset si nécessaire. """
    if not os.path.exists("./data"):
        os.makedirs("data")

    if not os.path.exists("./data/enron_spam_data.csv"):
        if not os.path.exists("./data/dataset_enron_annoted.zip"):
            print("Téléchargement du dataset...")
            try:
                urllib.request.urlretrieve(
                    "https://github.com/MWiechmann/enron_spam_data/raw/refs/heads/master/enron_spam_data.zip",
                    "./data/dataset_enron_annoted.zip"
                )
                print("Téléchargement réussi.")
            except:
                print("Téléchargement échoué")
                exit(1)

        print("Extraction du dataset...")
        with zipfile.ZipFile("./data/dataset_enron_annoted.zip", 'r') as zip_ref:
            zip_ref.extractall("./data/")
        print("Extraction réussie.")

    return "./data/enron_spam_data.csv"

def load_dataset():
    """ Charge le dataset en mémoire. """
    download_dataset()
    return pd.read_csv("./data/enron_spam_data.csv")

def get_dataset():
    """ Prépare les données d'entraînement et de test. """
    dataset = load_dataset()
    len_dataset = len(dataset)
    dataset = dataset.drop(columns=["Date"])
    dataset.rename(columns={"Subject": "subject", "Message": "body"}, inplace=True)
    dataset["ground_truth"] = dataset["Spam/Ham"].apply(lambda x: 0 if x == "ham" else 1)
    dataset = dataset.drop(columns="Spam/Ham")

    dataset_train = dataset.iloc[:int(len_dataset * 0.8)]
    dataset_test = dataset.iloc[int(len_dataset * 0.8):]

    dataset_train = dataset_train.sample(len(dataset_train), random_state=42)
    dataset_test = dataset_test.sample(len(dataset_test), random_state=42)

    return {"train": dataset_train, "test": dataset_test}

class ForestModel:
    """
    Modèle basé sur Random Forest pour détecter les spams.
    Entraîne le modèle à chaque exécution (pas besoin de fichier .pkl).
    """
    def __init__(self):
        """
        Initialise et entraîne le modèle Random Forest à chaque exécution.
        """
        print("Entraînement du modèle Random Forest en cours...")

        # Charger les données
        data = get_dataset()
        train_df = data["train"]

        # Extraire les features
        train_features = train_df.apply(lambda row: self.extraire_features(row["subject"], row["body"]), axis=1)
        self.train_features_df = pd.DataFrame(list(train_features))
        self.train_features_df["label"] = train_df["ground_truth"]

        # Séparer les données en train et test
        X_train, X_test, y_train, y_test = train_test_split(
            self.train_features_df.drop(columns=["label"]),
            self.train_features_df["label"],
            test_size=0.2,
            random_state=42,
            stratify=self.train_features_df["label"]
        )

        # Appliquer SMOTE si les spams sont rares
        if y_train.value_counts()[1] < 10:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Entraîner le modèle Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Évaluer la performance
        y_pred = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

    def extraire_features(self, subject, body):
        """
        Fonction pour extraire les caractéristiques d'un email.
        """
        mail = f"{subject} {body}".lower()
        mots = re.findall(r'\b\w+\b', mail)
        mots_freq = Counter(mots)

        # Calcul des métriques
        nombre_mots = len(mots)
        nombre_caracteres = len(mail)
        nombre_majuscules = sum(1 for mot in mots if mot.isupper())
        densite_majuscules = nombre_majuscules / nombre_mots if nombre_mots > 0 else 0
        densite_exclamations = mail.count('!') / nombre_mots if nombre_mots > 0 else 0
        densite_urls = len(URL_PATTERN.findall(mail)) / nombre_mots if nombre_mots > 0 else 0
        nombre_mots_spam = sum(1 for word in mots if word in SPAM_KEYWORDS)
        densite_spam_mots_clefs = nombre_mots_spam / nombre_mots if nombre_mots > 0 else 0

        return {
            "spam_mots_clefs": densite_spam_mots_clefs,
            "contient_url": bool(URL_PATTERN.search(mail)),
            "trop_exclamations": mail.count('!') > 5,
            "trop_majuscules": nombre_majuscules > nombre_mots * 0.3,
            "longueur_suspecte": nombre_caracteres > 5000,
            "nombre_mots": nombre_mots,
            "nombre_caracteres": nombre_caracteres,
            "densite_majuscules": densite_majuscules,
            "densite_exclamations": densite_exclamations,
            "densite_urls": densite_urls
        }

    def classify(self, mail):
        """
        Classifie un email en SPAM (1) ou NON SPAM (0).
        """
        features = self.extraire_features(mail["subject"], mail["body"])
        features_df = pd.DataFrame([features])
        prediction = self.model.predict(features_df)[0]
        return int(prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détection de spam basée sur Random Forest."
    )

    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Le sujet de l'email."
    )

    parser.add_argument(
        "--body",
        type=str,
        required=True,
        help="Le corps de l'email."
    )

    args = parser.parse_args()

    # Création et entraînement du modèle Random Forest
    model = ForestModel()

    # Création de l'email à analyser
    mail = {"subject": args.subject, "body": args.body}
    
    # Classification
    if model.classify(mail):
        print("Le mail est classé comme SPAM.")
        print(1)
    else:
        print("Le mail est classé comme NON SPAM.")
        print(0)


