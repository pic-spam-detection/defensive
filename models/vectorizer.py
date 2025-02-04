
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer
import joblib
import torch

class Vectorizer:

    def __init__(self, type='sklearn', checkpoint_path=None):

        if type == 'sklearn':
            self.sklearn(checkpoint_path)

        elif type == 'bert':
            self.bert()

    def sklearn(self, checkpoint_path=None):
        if checkpoint_path is not None:
            self.vectorizer = joblib.load(checkpoint_path + 'vectorizer_sklearn.joblib')
        else:
            self.vectorizer = CountVectorizer()

    def bert(self):
        # Charger le tokenizer BERT
        self.vectorizer = BertTokenizer.from_pretrained('bert-base-uncased')
