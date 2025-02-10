
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer
import joblib
import torch

class Vectorizer:

    def __init__(self, type='sklearn', checkpoint_path=None):
        self.type = type
        if type == 'sklearn':
            self.sklearn(checkpoint_path)

        elif type == 'bert':
            self.bert()
    
    def __call__(self, X):
        return self.vectorize(X)

    def sklearn(self, checkpoint_path=None):
        if checkpoint_path is not None:
            self.vectorizer = joblib.load(checkpoint_path + 'vectorizer_sklearn.joblib')
        else:
            self.vectorizer = CountVectorizer()

    def bert(self):
        # Charger le tokenizer BERT
        self.vectorizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def vectorize(self, X):
        if self.type == 'sklearn':
            return self.vectorizer.transform(X)

        elif self.type == 'bert':
            X = X.to_list()
            inputs = self.vectorizer(
                X,
                return_tensors='pt',
                truncation=True,
                max_length=10000,
                padding='max_length'
            )
            inputs = inputs['input_ids']
            inputs = inputs.reshape(inputs.shape[0], -1)
            return inputs
