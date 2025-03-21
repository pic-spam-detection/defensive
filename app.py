from flask import Flask, render_template, request, jsonify
import torch
import os
# Import your existing components - adjust imports based on your project structure

from src.dataset.dataloader import get_dataloader
from src.dataset.spam_dataset import SpamDataset
from src.models.classical_ml_classifier import ClassicalMLClassifier
from src.models.keywords_classifier import KeywordsClassifier
from src.models.ltsm_classifier import LSTM_classifier
from src.models.vectorizer import Vectorizer
from src.models.vote import Vote
from src.models.weight_manager import weight_manager
from src.models.roberta import Roberta
from src.test.classifier import test_classifier
from src.utils.results import Results

import json
import os
from typing import Literal, Optional

import click
import pandas as pd
import torch

app = Flask(__name__, 
           template_folder='webapp/templates',
           static_folder='webapp/static')

loaded_models = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    # Available options
    classifiers = ["naive_bayes", "logistic_regression", "svm", "keywords", "vote", "ltsm", "roberta"]
    vectorizers = ["sklearn", "bert"]
    
    prediction = None
    message = ""
    object = ""
    classifier = classifiers[0]  # Default selection
    vectorizer_type = vectorizers[0]  # Default selection
    vote_threshold = 0.5
    batch_size = 32
    use_meta_features = False
    
    if request.method == 'POST':
        # Get form data
        classifier = request.form.get('classifier')
        vectorizer_type = request.form.get('vectorizer')

        if classifier in loaded_models.keys():
            model = loaded_models[classifier]
        
        else:
            model = run_model(classifier, vote_threshold, device, "embeddings/train.pt", "embeddings/test.pt", None, batch_size, vectorizer_type, use_meta_features)

        message = request.form.get('message')
        object = request.form.get('object')

        mail = {"subject": object, "body": message}

        prediction = model.classify(mail)
        
        
        # Here you would process the message and get a prediction
        # prediction = process_message(message, vectorizer_manager, classifier)
        prediction = "The message is a spam" if prediction == 1 else "The message is not a spam"
        
    return render_template('index.html', 
                          classifiers=classifiers,
                          vectorizers=vectorizers,
                          prediction=prediction,
                          selected_classifier=classifier,
                          selected_vectorizer=vectorizer_type,
                          message=message,
                          object=object)

def run_model(classifier: str,
              vote_threshold: float,
              device: str,
              train_embeddings_path: Optional[str],
                test_embeddings_path: Optional[str],
                checkpoint_path: Optional[str],
              batch_size: Optional[int] = 32,
              vectorizer: Literal["sklearn", "bert"] = "sklearn",
              use_meta_features: Optional[bool] = False,
              vectorizer_trained = None,
              ):
    

    train_dataset = SpamDataset(split="train")
    test_dataset = SpamDataset(split="test")

    if classifier == "keywords":
        model = KeywordsClassifier()


    elif classifier == "roberta":
        model = Roberta()

    else:
        vectorizer_manager = Vectorizer(type=vectorizer)

        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")

        if vectorizer== "sklearn" or train_embeddings_path is None:
            train_embeddings, test_embeddings = vectorizer_manager(
                [sample["text"] for sample in train_dataset],
                [sample["text"] for sample in test_dataset],
                save_folder="embeddings",
            )

        else:
            train_embeddings = torch.load(train_embeddings_path, weights_only=False)
            test_embeddings = torch.load(test_embeddings_path, weights_only=False)


        test_labels = [sample["label"] for sample in test_dataset]
        test_dataloader = get_dataloader(test_embeddings, test_labels, batch_size)

        if classifier == "ltsm":
            input_dim = vectorizer_manager.get_vocab_size() + 1
            model = LSTM_classifier(input_dim, checkpoint_path)

        elif classifier == "vote":
            model = Vote(checkpoint_paths=checkpoint_path, threshold=vote_threshold, use_meta_features=use_meta_features)

        else:
            model = ClassicalMLClassifier(classifier, checkpoint_path, vectorizer_manager)

        if checkpoint_path is None:
            train_labels = [sample["label"] for sample in train_dataset]
            train_dataloader = get_dataloader(
                train_embeddings, train_labels, batch_size
            )
            model.train(train_dataloader)

    loaded_models[classifier] = model
    return model

    




if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else: 
        device = "cpu"

    app.run(debug=True)
