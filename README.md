# Building a Disaster Response Pipeline

## Project Overview
During and after natural disasters, disaster related messages and sns posts increase significantly. The main purpose of this project is to build a web app that can help emergency organizations analyze real-time messages and classify them into specific categories (e.g. Aid-related, hospitals, water). The model for this app was based on Nature Language Processing and Random Forest Classifier, with the data collected by Figure Eight. 

## File Description
**Data**
* [process_data.py](https://github.com/yyklee/disaster-response-pipeline/blob/main/data/process_data.py): Inputs csv files - disaster_messages.csv and disaster_categories.csv. Cleans and merge two datasets, and lastly creates a SQL database. 
* [disaster_messages.csv](https://github.com/yyklee/disaster-response-pipeline/blob/main/data/disaster_messages.csv): Raw messages data.
* [disaster_categories.csv](https://github.com/yyklee/disaster-response-pipeline/blob/main/data/disaster_categories.csv): Raw categories data. 

**Model**
* [train_classifier.py](https://github.com/yyklee/disaster-response-pipeline/blob/main/models/train_classifier.py): Python script to train model.

**App**
* [go.html](https://github.com/yyklee/disaster-response-pipeline/blob/main/app/templates/go.html)& [master.html](https://github.com/yyklee/disaster-response-pipeline/blob/main/app/templates/master.html): Scripts to create and start the Flask server.
* [run.py](https://github.com/yyklee/disaster-response-pipeline/blob/main/app/run.py): Python script needed to run the web app including models and graphs. 

## Instructions
1. Following command to run ETL pipeline that cleans data and stores in database python: data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 

3. Following command to run ML pipeline that trains classifier and saves python: models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

4. Finally, run the following command in the app's directory to run your web app: cd app -> python run.py


## Technologies
* HTML
* Python (pandas, numpy, scikit-learn, re, pickle, sys, sklearn, plotly, nltk, flask)

## Acknowledgements
This project is a part of the Udacity Data Science Nanodegree.

