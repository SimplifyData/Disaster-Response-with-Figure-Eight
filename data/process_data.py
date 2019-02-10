import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages =  pd.read_csv(message_filepath)
    categories = pd.read_csv(categories_filepath, sep = ",")
    df = pd.concat([messages, categories], axis = 1)
    return df


def clean_data(df):
    categories = df.categories.str.split(";", expand = True)
    
    #column names for categories
    category_colnames = list(map(lambda x : x[:-2] ,row))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        #print(column)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # drop the original categories column from `df`
    df.drop(["categories"],axis = 1,  inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_1 = pd.concat([df,categories], axis = 1)
    
    # drop duplicates
    df_1.drop("id", axis =1,inplace = True)
    
    # check number of duplicates
    df_1["id"] = messages.id
    
    return df_1


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    
    df.to_sql('Disaster_Response', engine, index=False, chunksize = 500) 


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