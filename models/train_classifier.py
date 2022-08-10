
# import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
nltk.download(['punkt', 'wordnet', 'omw-1.4','stopwords'])

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV


def load_data(data_file):
    '''Load and merge datasets'''
    # read in file
    engine = create_engine('sqlite:///' + data_file)
    df = pd.read_sql_table('message_category', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(np.array(y.columns))

    return X, y, category_names

def tokenize(text):
    '''
    tokenize, clean and lemmatize every row of message data'''
    
    #tokenize columns
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(text)

    #remove stop words from token list in each column
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    #lemmatize columns
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
    
def build_model():
    '''text processing and model pipeline, define
    parameters for gridsearch and create gridsearch object'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {'vect__min_df': [1, 5],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    inputs: model, X_test, y_test, category_names
    output: scores
    """
    y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col], y_pred[:, i], average='weighted')

        print('\nReport for the column ({}):\n')
        print('Precision: {}'.round(precision, 2))
        print('Recall: {}'.round(recall, 2))
        print('f-score: {}'.round(fscore, 2))
        

def export_model(model, model_filepath):
    '''Save model to a pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))


def run_pipeline():
    '''function that runs the whole pipeline: loading data to saving model'''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # build model pipeline
        print('Building model...')
        model = build_model()
        
        #train model pipeline
        print('Training model...')
        model.fit(X_train, y_train)
        
        #evaluate model pipeline
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        # save model
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        export_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    run_pipeline()