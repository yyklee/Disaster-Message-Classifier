
# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    category_colnames = list(row)

    # change the name of the columns of categories dataframe
    cat_col = []
    for i in category_colnames:
        col = i.split('-')[0]
        cat_col.append(col)
    
    categories.columns = cat_col

    # change the values in the categories dataframe to 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str.split('-').str[1]
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # replace categories colums in df with new category columns
    df.drop(columns= 'categories', inplace = True)
    df = pd.concat([df, categories], axis= 1)

    # remove duplicates
    df['is_duplicated'] = df.duplicated()
    df.drop_duplicates(inplace = True)

    #Drop the original categories column and is_duplicated columns from `df`
    #Remove child alone as it has all zeros only, 
    df.drop(columns= ['child_alone', 'is_duplicated'], inplace = True)

    # Replacing 2 with 1 in related column, as it has repeated amount among all other columns 
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('message_category', engine, index=False)


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