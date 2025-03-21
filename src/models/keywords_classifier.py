import argparse
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from torch.utils.data import DataLoader

from src.models.base_model import BaseModel


class KeywordsClassifier(BaseModel):
    def __init__(self):
        self.mots_clefs = [
            "free",
            "urgent",
            "offer",
            "gift",
            "promotion",
            "win",
            "money",
            "congratulations",
            "100%",
            "refund",
            "cash",
            "limited offer",
            "don't miss out",
            "sale",
            "discount",
            "satisfaction guaranteed",
            "exclusive",
            "best price",
            "activate",
            "password",
            "click here",
            "try",
            "free shipping",
            "loan",
            "fast",
            "benefit",
            "income",
            "extra income",
            "lottery",
            "jackpot",
            "VIP",
            "your account",
            "access",
            "secure",
            "unsolicited",
            "spam",
            "unsubscribe",
            "delete",
            "confidential",
            "verify",
            "payment",
            "invoice",
            "freebie",
            "sample",
            "today only",
            "limited",
            "exclusive promotion",
            "approval",
            "trial",
            "contact us",
            "get rich",
            "property",
            "bonus",
            "exclusive sale",
            "instant loan",
            "urgent",
            "100% free",
            "expired",
            "recover",
            "big",
            "growth",
            "immediate",
            "credit",
            "insurance",
            "crypto",
            "Bitcoin",
            "investment",
            "take advantage",
            "multiply",
            "cheap",
            "great deal",
            "work from home",
            "miracle",
            "no risk",
            "career",
            "job",
            "passive income",
            "prove",
            "start now",
            "share",
            "opportunity",
            "immediately",
            "great chance",
            "update",
            "renew",
            "gifts",
            "new customer",
            "secrets",
            "amazing",
            "very important",
            "only for you",
            "participate",
            "no cost",
            "instant payment",
            "registered",
            "for free",
            "cancellation",
            "confirm now",
        ]

    def classify(self, mail: Dict[str, str]) -> Tuple[int, bool]:
        """
        Vérifie si le mail contient des mots-clés associés au spam.
        :param mail: Dictionnaire contenant les informations du mail (adresse, sujet, corps, etc.)
        :return: 1 si spam, 0 sinon
        """
        full_mail = f"{mail['subject']} {mail['body']}".lower()
        pred = int(any(mot in full_mail for mot in self.mots_clefs))
        ground_truth = mail["ground_truth"]

        return pred, pred == ground_truth

    def train_model(self, _dataloader: DataLoader, _save_path: Optional[str] = None):
        # no training is done
        pass

    def get_model(self):
        # no model for this method
        return None

    def predict(self, X: Union[List[str], pd.Series]) -> List[int]:
        results = []
        for email in X:
            pred = int(any(mot in email.split(" ") for mot in self.mots_clefs))
            results.append(pred)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détection de spam basée sur des mots-clés."
    )

    parser.add_argument(
        "--subject", type=str, required=True, help="Le sujet de l'email."
    )

    parser.add_argument("--body", type=str, required=True, help="Le corps de l'email.")

    args = parser.parse_args()

    mail = {"subject": args.subject, "body": args.body}
    model = KeywordsClassifier()
    is_spam, _ = model.classify(mail)

    if is_spam:
        print("Le mail contient des mots suspectés de spam.")
        print(1)
    else:
        print("Le mail ne contient aucun mot suspecté de spam.")
        print(0)
