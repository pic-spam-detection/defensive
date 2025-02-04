from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from models.vectorizer import Vectorizer


class NaiveBayes:
    
    def __init__(self, dataset, checkpoint_path=None):
        self.classifier = MultinomialNB()
        self.vectorizer = Vectorizer(checkpoint_path=checkpoint_path).vectorizer

        if checkpoint_path is not None:
            self.classifier = joblib.load(checkpoint_path + 'classifier.joblib')
        
    def train(self, dataset, save_path=None):

        mails_df = pd.DataFrame(dataset)
        mails_df['text'] = mails_df['subject'] + ' ' + mails_df['body']

        y = mails_df['ground_truth']

        X_train, X_test, y_train, y_test = train_test_split(mails_df['text'], y, test_size=0.2, random_state=47)

        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)
        self.classifier.fit(X_train, y_train)

        if save_path is not None:
            joblib.dump(self.classifier, save_path + 'classifier.joblib')
            joblib.dump(self.vectorizer, save_path + 'vectorizer.joblib')

        y_pred = self.classifier.predict(X_test)

        accuracy = (y_pred == y_test).mean()
        print(f"Accuracy: {accuracy} {accuracy_score(y_test, y_pred)}")
        return accuracy

    def classify(self, mail):
        """
        args:
            mail: dict
                mail information (adress, domain, domain_extension, subject, body, ground_truth)

        returns:
            (boolean) prediction 0 if ham, 1 if spam, (boolean) prediction == ground_truth
        """
        try:
            adress = mail['adress']
            domain = mail['domain']
            domain_extension = mail['domain_extension']
            subject = mail['subject']
            body = mail['body']
            ground_truth = mail['ground_truth']
        except:
            raise ValueError("mail should be dictionary like with 'adress', 'domain', 'domain_extension', 'subject', 'body' and 'ground_truth'")


        text =  mail['subject'] + ' ' + mail['body']
        X = self.vectorizer.transform(text)
        pred = self.classifier.predict(X)

        return pred, pred==ground_truth


