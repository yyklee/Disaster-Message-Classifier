
# import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
nltk.download(['punkt', 'wordnet', 'omw-1.4','stopwords', 'averaged_perceptron_tagger'])

# sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



def load_data(data_file):
    '''Load and merge datasets'''
    # read in file
    engine = create_engine('sqlite:///' + data_file)
    df = pd.read_sql_table('message_category', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(np.array(y.columns))

    return X, y, category_names

def tokenize(text, url_place_holder_string = "urlplaceholder"):
    '''
    Function that tokenizes the text data. 

    Input:
        text - Text messages that need tokenizing
    Output:
        clean_tokens - List of tokens extracted from the original text
    '''
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the input text
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extracting word tokens from the input text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(text)
    
    # Remove stop words from token list in each column
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatizer to standardize words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    '''
    Pipeline function that processes text, define algorithms and
    related parameters for gridsearch and create gridsearch object
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

        ])),

        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'classifier__estimator__n_estimators': [10,25]}

    
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose =3)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Inputs: model, X_test, y_test, category_names
    Output: scores
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test.values, y_pred, target_names= category_names))
    
        

def export_model(model, model_filepath):
    '''Save model to a pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))


def run_pipeline():
    '''Function that runs the whole pipeline: loading data to saving model'''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath) 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # build model pipeline
        print('Building model...')
        model = build_model()
        
        #train model pipeline
        print('Training model...')
        model.fit(X_train, Y_train)
        
        #evaluate model pipeline
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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