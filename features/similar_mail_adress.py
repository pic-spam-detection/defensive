import sys
!pip install python-Levenshtein
from Levenshtein import distance


def verifier_email_similaire(email, email_cible="service-client@carrefour.fr"):
    """
    Vérifie si une adresse e-mail est similaire à l'adresse cible à un ou deux caractères près.
    
    :param email: L'adresse e-mail à vérifier
    :param email_cible: L'adresse e-mail de référence
    :return: True si similaire, False sinon
    """
    # Calcul de la distance de Levenshtein
    dist = distance(email, email_cible)
    
    # Vérification de la similarité (distance <= 2)
    if dist <= 2:
        print(f"L'adresse '{email}' est SIMILAIRE à '{email_cible}' (distance : {dist}).")
        return True
    else:
        print(f"L'adresse '{email}' est DIFFERENTE de '{email_cible}' (distance : {dist}).")
        return False


adresse_email = "service-clinet@carrefour.fr"  # Exemple d'e-mail à tester
verifier_email_similaire(adresse_email)
