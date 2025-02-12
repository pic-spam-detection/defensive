# 🛡️ Équipe Défense : Détection de Spams  

## 📖 **Objectif**  
Ce repository contient les travaux de l’équipe Défense pour :  
- Développer un modèle de classification capable de **détecter les spams**.  
- Évaluer et comparer notre solution avec les outils existants.  


## **Installation**

Cloner le dépôt `git`:

```bash
git clone https://github.com/pic-spam-detection/defensive.git
cd defensive
```

Installer les librairies nécessaires :

```bash
pip install -r requirements.txt
```

## **Usage**

```bash
python3 -m spam_detection.py
```


## **Code usage**

Pour charger le dataset : \
_Pour simplement charger la version raw du dataset, cf utils/dataset.py_


```python
from utils.dataset import get_dataset

data = get_dataset()
```

data : dictionnaire {"train": pandas dataframe, "test": pandas dataframe}

Format de train et set :

| Message ID | subject | body | address | domain | domain_extension | ground_truth |
|------------|---------|------|---------|--------|------------------|--------------|
| 8280       | this is a ham | contenu du mail | test_user | gmail | com | 0 |
| 14425      | this is a spam | contenu du mail | test_user.spammer | gmail | com | 1 |


À la première utilisation, le modèle se chargera, ce qui prend quelques minutes.
