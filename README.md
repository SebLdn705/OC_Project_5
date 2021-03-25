# OC_Project_5
OC Project with the aim to create an auto tag suggestion based on ML Technics. The query is made via a Flask API.


##**Table of contents**

* [General Info](#general-info)
* [Technologies](#technologies)
* [How to install](#how-to-install)
* [ML Model](#ML_Model_and_assumptions_/_limitations_of_the_model)

## General Info

The API retrieves the most likely tags to put according to the question/title of a stackoverflow query. 
It returns a JSON file containing the tags.

### ML Model and assumptions / limitations of the model

The model is based on a supervised model using the SVM SVC and training with a f1-micro scoring. The possible number of
tags was limited to the top 100 tags used. If no tag is retrieved it is maybe best to try to reformulate the question in
a better manner by using more keywords (programming specific language)


### How to install

Clone the repository using git clone https://github.com/SebLdn705/OC_Project_5.git

In order to run the model from a jupyter notebook, you'll need to run the python api.py. The server should be working 
http://127.0.0.1:5000/predict

### How to pass the question to the API

The API requires the question to be sent under a JSON format. For instance, to get the predicted tags it is recommended 
to send the question with the below format (example from a Jupyter Notebook)

import requests

URL = 'http://127.0.0.1.5000/predict'

params = {'question': 'Your stack overflow question/Title'} -> format is under a dictionary type

r = requests.post(url=URL, json=params)

question = r.json()

### Technologies

The API is running with:
* nltk: 3.5
* flask: 1.1.2
* joblib: 0.14.1
* pandas: 1.0.3




