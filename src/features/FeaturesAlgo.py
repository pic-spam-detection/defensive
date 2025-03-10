import argparse
import re
from collections import Counter

# Expressions régulières et mots-clés de spam
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
SPAM_KEYWORDS = {
    "free", "win", "money", "prize", "buy now", "urgent", "offer", "gift",
    "promotion", "congratulations", "100%", "refund", "cash", "limited offer"
}

class Features:
    """
    Modèle basé sur l'analyse des caractéristiques d'un email
    (exclamations, majuscules, URLs, longueur, etc.).
    """
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

        # Calcul des métriques
        nombre_mots = len(mots)
        nombre_caracteres = len(mail_content)
        nombre_majuscules = sum(1 for mot in mots if mot.isupper())
        densite_majuscules = nombre_majuscules / nombre_mots if nombre_mots > 0 else 0
        densite_exclamations = mail_content.count('!') / nombre_mots if nombre_mots > 0 else 0
        densite_urls = len(URL_PATTERN.findall(mail_content)) / nombre_mots if nombre_mots > 0 else 0
        nombre_mots_spam = sum(1 for word in mots if word in SPAM_KEYWORDS)
        densite_spam_mots_clefs = nombre_mots_spam / nombre_mots if nombre_mots > 0 else 0

        # Création du dictionnaire des features
        features = {
            "spam_mots_clefs": densite_spam_mots_clefs,
            "contient_url": bool(URL_PATTERN.search(mail_content)),
            "trop_exclamations": mail_content.count('!') > 3,
            "trop_majuscules": nombre_majuscules > nombre_mots * 0.3,
            "longueur_suspecte": nombre_caracteres > 5000,
            "nombre_mots": nombre_mots,
            "nombre_caracteres": nombre_caracteres,
            "densite_majuscules": densite_majuscules,
            "densite_exclamations": densite_exclamations,
            "densite_urls": densite_urls
        }

        return int(self.evaluer_spam(features) == "SPAM")

    def evaluer_spam(self, features):
        """
        Évalue si un email est un spam basé sur un score de règles.
        """
        score = 0

        if features["spam_mots_clefs"] > 0.02:
            score += 2
        if features["contient_url"]:
            score += 1
        if features["trop_exclamations"]:
            score += 2
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

        return "SPAM" if score >= 3 else "NON SPAM"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détection de spam basée sur les caractéristiques textuelles."
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
    
    mail = {"subject": args.subject, "body": args.body}
    model = Features()
    
    if model.classify(mail):
        print("Le mail est classé comme SPAM.")
        print(1)
    else:
        print("Le mail est classé comme NON SPAM.")
        print(0)


