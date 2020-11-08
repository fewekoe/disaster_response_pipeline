# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
import pickle


import re
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Load data into pandas DataFrames

    Arguments:
        the database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize function

    Arguments:
        text -> messages in english
    Output:
        clean_tokens -> tokenized text, clean for ML modeling
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class based on exercises on Udacity

    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {'clf__estimator__n_estimators': [50, 100, 200],
                  'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
                  'features__text_pipeline__vect__max_df': (0.75, 1.0),
                  'features__text_pipeline__vect__max_features': (None, 5000),
                  'features__text_pipeline__tfidf__use_idf': (True, False),
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']#,
                  #'features__transformer_weights': (
                  #  {'text_pipeline': 1.0, 'starting_verb': 0.5},
                  #  {'text_pipeline': 0.5, 'starting_verb': 1.0},
                  #  {'text_pipeline': 0.8, 'starting_verb': 1.0},
                  #)
                   }

    #cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function

    This function applies ML pipeline to a test set and prints out
    model performance (accuracy and f1score)

    Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names (multi-output)
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))



def save_model(model, model_filepath):
    """
    Saving the model to a picke file

    Arguments:
        model -> GridSearchCV or Scikit Pipelin object

    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


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
