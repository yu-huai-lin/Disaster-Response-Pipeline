# import libraries
import pandas as pd
import numpy as np
import sys



def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv("disaster_messages.csv")
    # load categories dataset
    categories = pd.read_csv("disaster_categories.csv")
    
    # merge datasets
    df = pd.merge(messages, categories, on=["id"])


    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    row = row.apply(lambda x: x[:-2]).tolist()
    
    # use this row to extract a list of new column names for categories.
    
    # rename the columns of `categories`
    categories.columns = row
    
    for column in categories:
    # set each value to be the last character of the string
    categories[column ] = categories[column ].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df =  pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates(keep=False)

    return df


def save_data(df, database_filename):
    from sqlalchemy import create_engine
    engine = create_engine(database_filename)
    df.to_sql('df, engine, index=False)
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()