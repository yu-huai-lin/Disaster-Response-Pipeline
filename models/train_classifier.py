import sys
from sqlalchemy import create_engine
import pickle
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

category_names = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']


def load_data(database_filepath, category_names):
    """
    Function: Load the prepared table as input data source
    Args:
      database_filepath(str): Path to the database
      category_names(list): List of the column names
    Return:
      X,Y(dataframes): Two dataframes, served as inputs and outputs of the ML model
    """
    #engine = create_engine(database_filepath)
    #engine=create_engine('sqlite:///data/DisasterResponse.db')
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('df', engine)  
    
    #Define target input and output
    X = df['message']
    
    Y = df[category_names]
    
    return X, Y

def tokenize(text):
    """
    Function: Clean, prepare, tokenize and lemmatize the text data for modelling
    Args:
      text: List of messages
    Return:
      tokens: A list of messages after processsing
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
     Function: Build the model pipelines
     Return:
       cv: The model for training
     """
    lr  = RandomForestClassifier()

    # build pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ("clf",MultiOutputClassifier(lr))])
    
    parameters = {
          'clf__estimator__min_samples_split': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    print(pipeline.get_params().keys())
    
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: Evaluate the model performance by looking at accuracy score.
    model: model
    X_test:Test data
    y_test:Test labels
    category_names: List of the column names
    """
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()

    print("Accuracy:", accuracy)
    
    # Calculate classification report
    report = classification_report(Y_test, y_pred, target_names=category_names)

    # Print the classification report
    print("Classification Report:\n", report)


def save_model(model, model_filepath):
    """
    Function: Save the model as a pickle file
    model:model
    model_filepath:Path of the pickle file
    """
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath, category_names )
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