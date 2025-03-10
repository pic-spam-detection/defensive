import argparse
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
#from utils.dataset import get_dataset  
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
# Télécharger les stopwords français 
nltk.download("stopwords")
stop_words = set(stopwords.words("french"))  

# Expressions régulières et mots-clés spam
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
SPAM_KEYWORDS = {"gratuit", "gagner", "argent", "cadeau", "offre spéciale", "urgent", "promotion", "remboursement"}

class TFIDFModel:
    """
    Modèle basé sur TF-IDF pour détecter le spam dans les emails.
    Entraîne le modèle à chaque exécution (pas besoin de fichier .pkl).
    """
    def __init__(self):
        """
        Initialise et entraîne le modèle TF-IDF à chaque exécution.
        """
        print("Entraînement du modèle TF-IDF en cours...")

        # Charger les données
        data = get_dataset()
        train_df = data["train"]

        # Entraîner le modèle TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=list(stop_words))
        self.vectorizer.fit(train_df["subject"].fillna("") + " " + train_df["body"].fillna(""))

        print("Modèle TF-IDF entraîné avec succès !")

    def classify(self, mail):
        """
        Analyse un email et retourne 1 (SPAM) ou 0 (NON SPAM).
        :param mail: Dictionnaire contenant 'subject' et 'body' de l'email.
        :return: 1 si spam, 0 sinon.
        """
        subject, body = mail["subject"], mail["body"]
        mail_content = f"{subject} {body}".lower()
        mots = re.findall(r'\b\w+\b', mail_content)
        mots_freq = Counter(mots)

        # Calcul des métriques classiques
        nombre_mots = len(mots)
        nombre_caracteres = len(mail_content)
        nombre_majuscules = sum(1 for mot in mots if mot.isupper())
        densite_majuscules = nombre_majuscules / nombre_mots if nombre_mots > 0 else 0
        densite_exclamations = mail_content.count('!') / nombre_mots if nombre_mots > 0 else 0
        densite_urls = len(URL_PATTERN.findall(mail_content)) / nombre_mots if nombre_mots > 0 else 0
        nombre_mots_spam = sum(1 for word in mots if word in SPAM_KEYWORDS)
        densite_spam_mots_clefs = nombre_mots_spam / nombre_mots if nombre_mots > 0 else 0

        # Calcul du score TF-IDF
        try:
            tfidf_score = np.mean(self.vectorizer.transform([mail_content]).toarray()) if nombre_mots > 0 else 0
        except Exception:
            tfidf_score = 0  # Si le modèle TF-IDF ne reconnaît pas de mots utiles

        # Création du dictionnaire des features
        features = {
            "spam_mots_clefs": densite_spam_mots_clefs,
            "contient_url": bool(URL_PATTERN.search(mail_content)),
            "trop_exclamations": mail_content.count('!') > 5,
            "trop_majuscules": nombre_majuscules > nombre_mots * 0.3,
            "longueur_suspecte": nombre_caracteres > 5000,
            "nombre_mots": nombre_mots,
            "nombre_caracteres": nombre_caracteres,
            "densite_majuscules": densite_majuscules,
            "densite_exclamations": densite_exclamations,
            "densite_urls": densite_urls,
            "tfidf_moyen": tfidf_score
        }

        return int(self.evaluer_spam(features) == "SPAM")

    def evaluer_spam(self, features):
        """
        Évalue si un email est un spam basé sur un score hybride TF-IDF + heuristiques.
        """
        score = 0

        if features["spam_mots_clefs"] > 0.02:
            score += 2
        if features["contient_url"]:
            score += 1
        if features["trop_exclamations"]:
            score += 1
        if features["trop_majuscules"]:
            score += 1
        if features["longueur_suspecte"]:
            score += 2
        if features["densite_majuscules"] > 0.2:
            score += 1
        if features["densite_exclamations"] > 0.1:
            score += 1
        if features["densite_urls"] > 0.05:
            score += 1
        if features["tfidf_moyen"] > 0.1:
            score += 2  # Plus le score TF-IDF est élevé, plus l'email est suspect

        return "SPAM" if score >= 3 else "NON SPAM"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détection de spam basée sur TF-IDF et heuristiques."
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

    # Création et entraînement du modèle TF-IDF
    model = TFIDFModel()

    # Création de l'email à analyser
    mail = {"subject": args.subject, "body": args.body}
    
    # Classification
    if model.classify(mail):
        print("Le mail est classé comme SPAM.")
        print(1)
    else:
        print("Le mail est classé comme NON SPAM.")
        print(0)
