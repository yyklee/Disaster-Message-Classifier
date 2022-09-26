# Building a Disaster Response Pipeline

## Project Overview
During and after natural disasters, . The main purpose of this project is to build a web app that can help emergency organizations analyze real-time messages and classify them into specific categories (e.g. Water, Food, Hospitals, Aid-Related). The model for this app was based on Nature Language Processing and Random Forest Classifier, with the data collected by Figure Eight. 

## File Description
**Data**
* process_data.py:
* disaster_messages.csv:
* disaster_categories.csv:

**Model**
* train_classifier.py:

**App**
* disaster_categories.csv:
* disaster_categories.html:

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

2. To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/ Or Go to http://localhost:3001/

## Technologies
* HTML
* Python (pandas, numpy, re, pickle, sys, sklearn, plotly, json, nltk, flask)

## Licensing, Authors, Acknowledgements
This project is a part of the Udacity Data Science Nanodegree.

