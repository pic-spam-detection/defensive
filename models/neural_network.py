from utils import dataset
import pandas as pd

data = get_dataset()
#Data est un dictionnaire avec les clefs "train" et "test". Train est un pandas, test aussi.

dataframe = pd.read_csv("../data/enron_spam_data.csv")
data_train = data["train"]
data_test = data["test"]

print("Colonnes de data_train :", data_train.columns)
print("\nPremières lignes de data_train :")
print(data_train.head())