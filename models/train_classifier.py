import sys
import pandas as pd
import numpy as np

import pickle

from sqlalchemy import create_engine

import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('words')

from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, recall_score, precision_score

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV,cross_val_score, cross_validate
from sklearn.metrics import fbeta_score, make_scorer, SCORERS


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filename))

    df = pd.read_sql("SELECT * from Disaster_Response", engine)

    X = df["message"].values

    Y = (df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
             'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter',
             'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
             'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools',
             'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
             'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
             'direct_report']].values)

    return X, Y

def tokenize(text):
    # normalizing all the text
    text = text.lower()

    # removing extra characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenizing all the sentences
    words = word_tokenize(text)

    # removing stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]

    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in stemmed]

    return lemmed


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    parameters = {}
    parameters["clf__estimator__oob_score"] = [True]
    parameters["clf__estimator__n_estimators"] = [100]

    grid_obj = GridSearchCV(pipeline, parameters)

    return grid_obj



def evaluate_model(model, X_test, Y_test, category_names):
    best_predictions =  model.predict(X_test)

    for i, col in enumerate(category_names):
        print(col)
        accuracy = accuracy_score(Y_test[i], best_predictions[i])
        precision = precision_score(Y_test[i], best_predictions[i])
        recall = recall_score(Y_test[i], best_predictions[i])
        print("\tAccuracy: %.4f\tPrecision: %.4f\t Recall: %.4f\n" % (accuracy, precision, recall))




def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()