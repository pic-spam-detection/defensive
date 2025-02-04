
from sklearn.feature_extraction.text import CountVectorizer
import joblib


class Vectorizer:

    def __init__(self, checkpoint_path=None):
        if checkpoint_path is not None:
            self.classifier = joblib.load(checkpoint_path + 'classifier.joblib')
            self.vectorizer = joblib.load(checkpoint_path + 'vectorizer.joblib')
        else:
            self.vectorizer = CountVectorizer(stop_words="english")

