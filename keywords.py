def demander_et_retourner():
    # Demander d'entrer le mail.
    entree = input("Veuillez écrire un mail : ")
    # Retourner l'entrée
    return entree

# Liste des mots-clés à vérifier
mots_clefs = [
    "gratuit", "urgent", "offre", "cadeau", "promotion", "gagner", "argent", 
    "félicitation", "100%", "remboursement", "cash", "offre limitée", 
    "ne manquez pas", "solde", "réduction", "satisfaction garantie", 
    "exclusif", "meilleur prix", "activer", "mot de passe", "password", 
    "cliquez ici", "essayez", "livraison gratuite", "prêt", "rapide", 
    "bénéfice", "revenu", "revenu supplémentaire", "loterie", "jackpot", 
    "VIP", "votre compte", "accès", "sécurisé", "non sollicité", "spam", 
    "désabonnement", "supprimer", "confidentiel", "vérifiez", 
    "paiement", "facture", "gratuité", "échantillon", "aujourd'hui seulement", 
    "limité", "promotion exclusive", "approbation", "essai", "contactez-nous", 
    "devenez riche", "propriété", "bonus", "solde exclusif", 
    "prêt instantané", "urgent", "100% gratuit", "expiré", "récupérer", 
    "gros", "croissance", "immédiat", "crédit", "assurance", "crypto", 
    "Bitcoin", "investissement", "profitez", "multiplier", "pas cher", 
    "super affaire", "travail à domicile", "miracle", "aucun risque", 
    "carrière", "emploi", "revenu passif", "prouvez", "démarrer maintenant", 
    "partager", "opportunité", "immédiatement", "grande chance", 
    "mise à jour", "renouveler", "cadeaux", "nouveau client", "secrets", 
    "formidable", "très important", "seulement pour vous", "participer", 
    "aucun coût", "paiement immédiat", "enregistré", "gratuitement", 
    "annulation", "confirmez maintenant"
]


def detection_spam_mots_clefs(mail):
    # Initialisation de la variable pour indiquer si c'est un spam
    is_spam = False
    
    # Vérification des mots-clés dans le texte
    for mot in mots_clefs:
        if mot in mail.lower():  # Vérifie si le mot-clé est présent (insensible à la casse)
            is_spam = True
            break  # Sortir de la boucle dès qu'un mot-clé est trouvé
    
    return is_spam

def verifier_urls(mail):
    mots = mail.split()
    url_keywords = ["http", "www", ".com", ".net", ".org"]
    return any(url_keyword in mot for mot in mots for url_keyword in url_keywords)
def verifier_exclamations(mail):
    return mail.count('!') > 5  
# Exemple d'utilisation
mail = demander_et_retourner()
if detection_spam_mots_clefs(mail):
    print("Le mail contient des mots suspectés de spam.")
    print(1)
else:
    print("Le mail ne contient aucun mot suspecté de spam.")
    print(0)

if verifier_urls(mail):
    print("Le mail contient des liens ou des URL suspectes.")
    print(1)
if verifier_exclamations(mail):
    print("Le mail contient trop de points d'exclamation, ce qui est suspect.")
    print(1)
