from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from models.vectorizer import Vectorizer


class NaiveBayes:
    
    def __init__(self, dataset, type='sklearn', checkpoint_path=None):
        self.classifier = MultinomialNB()
        self.vectorizer = Vectorizer(type=type, checkpoint_path=checkpoint_path).vectorizer
        self.type = type

    def train(self, dataset, save_path=None):

        mails_df = pd.DataFrame(dataset)
        mails_df['text'] = mails_df['subject'] + ' ' + mails_df['body']


        y = mails_df['ground_truth']

        X_train, X_test, y_train, y_test = train_test_split(mails_df['text'], y, test_size=0.2, random_state=47)

        if self.type == 'bert':
            X_train = X_train.to_list()
            X_test = X_test.to_list()

            inputs_train = self.vectorizer(X_train, return_tensors='np', truncation=True, max_length=10000, padding="max_length")
            inputs_test = self.vectorizer(X_test, return_tensors='np', truncation=True, max_length=10000, padding="max_length")

            X_train_ids = inputs_train['input_ids']
            X_test_ids = inputs_test['input_ids']

            # Flatten the arrays to 2D
            X_train_ids = X_train_ids.reshape(X_train_ids.shape[0], -1)
            X_test_ids = X_test_ids.reshape(X_test_ids.shape[0], -1)


            self.classifier.fit(X_train_ids, y_train)
            y_pred = self.classifier.predict(X_test_ids)

        elif self.type == 'sklearn':

            X_train = self.vectorizer.fit_transform(X_train)
            X_test = self.vectorizer.transform(X_test)

            self.classifier.fit(X_train, y_train)


            if save_path is not None:
                joblib.dump(self.classifier, save_path + 'classifier_bayes.joblib')
                joblib.dump(self.vectorizer, save_path + 'vectorizer_sklearn.joblib')

            y_pred = self.classifier.predict(X_test)
            
        accuracy = (y_pred == y_test).mean()
        print(f"Accuracy: {accuracy} {accuracy_score(y_test, y_pred)}")
        return accuracy

    def classify(self, mail):
        """
        args:
            mail: dict
                mail information (address, domain, domain_extension, subject, body, ground_truth)

        returns:
            (boolean) prediction 0 if ham, 1 if spam, (boolean) prediction == ground_truth
        """
        try:
            adress = mail['address']
            domain = mail['domain']
            domain_extension = mail['domain_extension']
            subject = mail['subject']
            body = mail['body']
            ground_truth = mail['ground_truth']
        except:
            raise ValueError("Mail should be dictionary like 'address', 'domain', 'domain_extension', 'subject', 'body' and 'ground_truth'")


        text =  mail['subject'] + ' ' + mail['body']

        if self.type == 'bert':
            X = self.vectorizer(text.to_list(), return_tensors='np', truncation=True, max_length=10000, padding="max_length")['input_ids']
        elif self.type == 'sklearn':
            X = self.vectorizer.transform(text)
        pred = self.classifier.predict(X)

        return pred, pred==ground_truth


