# Building a Disaster Response Pipeline

## Project Overview
During and after natural disasters, disaster related messages and sns posts increase significantly. The main purpose of this project is to build a web app that can help emergency organizations analyze real-time messages and classify them into specific categories (e.g. Water, Food, Hospitals, Aid-Related). The model for this app was based on Nature Language Processing and Random Forest Classifier, with the data collected by Figure Eight. 

## File Description
**Data**
* [process_data.py](https://github.com/yyklee/disaster-response-pipeline/blob/main/data/process_data.py): Inputs csv files (message data and message categories data), clean and merge two datasets, and lastly creates a SQL database. 
* [disaster_messages.csv](https://github.com/yyklee/disaster-response-pipeline/blob/main/data/disaster_messages.csv): Raw messages data.
* [disaster_categories.csv](https://github.com/yyklee/disaster-response-pipeline/blob/main/data/disaster_categories.csv): Raw categories data. 

**Model**
* [train_classifier.py](https://github.com/yyklee/disaster-response-pipeline/blob/main/models/train_classifier.py): Python script to train model.

**App**
* [go.html](https://github.com/yyklee/disaster-response-pipeline/blob/main/app/templates/go.html)& [master.html](https://github.com/yyklee/disaster-response-pipeline/blob/main/app/templates/master.html): Scripts to create and start the Flask server.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

2. To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

3. To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

4. Go to http://0.0.0.0:3001/ Or Go to http://localhost:3001/

## Technologies
* HTML
* Python (pandas, numpy, re, pickle, sys, sklearn, plotly, json, nltk, flask)

## Acknowledgements
This project is a part of the Udacity Data Science Nanodegree.

