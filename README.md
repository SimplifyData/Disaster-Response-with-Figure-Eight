# Disaster Response Pipeline with Figure Eight 

Apply Data Engineering to Build ETL &amp; NLP Pipelines and Create an App for Disaster Relief using Flask 

In this project, I am applying Data Engineering & Data Science skills to analyze disaster data from Figure Eight to 
build a model for an API that classifies disaster messages.

The project contains data set containing real messages that were sent during disaster events. 
I am creating a machine learning pipeline to categorize these events so that it can be sent to an appropriate 
disaster relief agency.

Project includes a web app where an emergency worker can input a new message and get classification results 
in several categories. The web app displays visualizations of the data.

### There are three components in this project.

1) ETL Pipeline
A Python script, process_data.py, a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2) ML Pipeline
A Python script, train_classifier.py, a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3) Flask Web App
