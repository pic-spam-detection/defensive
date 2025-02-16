import urllib.request
import zipfile
import os
import pandas as pd

def download_dataset():
    if not os.path.exists("./data"):
        os.makedirs("data")

    if not os.path.exists("./data/enron_spam_data.csv"):
        if not os.path.exists("./data/dataset_enron_annoted.zip"):
            print("Téléchargement du dataset...")
            try:
                urllib.request.urlretrieve("https://github.com/MWiechmann/enron_spam_data/raw/refs/heads/master/enron_spam_data.zip", 
                                "./data/dataset_enron_annoted.zip")
                print("Téléchargement réussi.")
            except:
                print("Téléchargement échoué")
                exit(1)

        if not os.path.exists("./data/enron_spam_data.csv"):
            print("Extraction du dataset")
            with zipfile.ZipFile("./data/dataset_enron_annoted.zip", 'r') as zip_ref:
                zip_ref.extractall("./data/")
            print("Extraction réussie.")

    return "./data/enron_spam_data.csv"

def load_dataset():
    download_dataset()
    return pd.read_csv("./data/enron_spam_data.csv")

def get_dataset():
    dataset = load_dataset()
    len_dataset = len(dataset)
    dataset = dataset.drop(columns=["Date"])
    dataset["adress"] = ""
    dataset["domain"] = ""
    dataset["domain_extension"] = ""
    dataset.rename(columns={"Subject": "subject", "Message": "body"}, inplace=True)
    dataset["ground_truth"] = dataset["Spam/Ham"].apply(lambda x: 0 if x == "ham" else 1)
    dataset = dataset.drop(columns="Spam/Ham")

    dataset_train = dataset.iloc[:int(len_dataset * 0.8)]
    dataset_test = dataset.iloc[int(len_dataset * 0.8):]

    dataset_train = dataset_train.sample(len(dataset_train), random_state=42)
    dataset_test = dataset_test.sample(len(dataset_test), random_state=42)

    dataset = {"train": dataset_train, "test": dataset_test}

    return dataset









