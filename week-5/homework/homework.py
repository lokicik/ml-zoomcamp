import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer



# Question 1
# Install Pipenv
# What's the version of pipenv you installed?
# Use --version to find out

!pip install pipenv
!pipenv --version # pipenv, version 2023.10.3

# Question 2
# Use Pipenv to install Scikit-Learn version 1.3.1
# What's the first hash for scikit-learn you get in Pipfile.lock?
# Note: you should create an empty folder for homework and do it there.

!pipenv install scikit-learn==1.3.1
# 0c275a06c5190c5ce00af0acbb61c06374087949f643ef32d355ece12c4db043


# Question 3
# Let's use these models!
# Write a script for loading these models with pickle
# Score this client:
# {"job": "retired", "duration": 445, "poutcome": "success"}
# What's the probability that this client will get a credit?

import pickle
input_model = 'week-5/homework/model1.bin'

with open(input_model, 'rb') as f_in:
    model = pickle.load(f_in)
model

input_dv = "week-5/homework/dv.bin"
with open(input_dv, 'rb') as f_in:
    dv = pickle.load(f_in)
dv

customer = {"job": "retired", "duration": 445, "poutcome": "success"}
X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]
print('input:', customer)
print('output:', y_pred)
# 0.9019309332297606

# Question 4
# Now let's serve this model as a web service
# Install Flask and gunicorn (or waitress, if you're on Windows)
# Write Flask code for serving the model
# Now score this client using requests:
# url = "YOUR_URL"
# client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
# requests.post(url, json=client).json()
# What's the probability that this client will get a credit?

# run python predicthomework.py on terminal first
import requests
url = "http://localhost:9696/predicthomework"
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
requests.post(url, json=client).json()
# Out[43]: {'poutcome': False, 'poutcome_probability': 0.13968947052356817}


# Docker
# Install Docker. We will use it for the next two questions.
# For these questions, we prepared a base image: svizor/zoomcamp-model:3.10.12-slim. You'll need to use it (see Question 5 for an example).
# This image is based on python:3.10.12-slim and has a logistic regression model (a different one) as well a dictionary vectorizer inside.
# This is how the Dockerfile for this image looks like:
# FROM python:3.10.12-slim
# WORKDIR /app
# COPY ["model2.bin", "dv.bin", "./"]
# We already built it and then pushed it to svizor/zoomcamp-model:3.10.12-slim.
# Note: You don't need to build this docker image, it's just for your reference.



# Question 5
# Download the base image svizor/zoomcamp-model:3.10.12-slim. You can easily make it by using docker pull command.
# So what's the size of this base image?
# 47 MB
# 147 MB
# 374 MB
# 574 MB
# You can get this information when running docker images - it'll be in the "SIZE" column.


# docker pull svizor/zoomcamp-model:3.10.12-slim
# docker image inspect svizor/zoomcamp-model:3.10.12-slim --format='{{.Size}}'
# 147133320 bytes




# Dockerfile
# Now create your own Dockerfile based on the image we prepared.
# It should start like that:
# FROM svizor/zoomcamp-model:3.10.12-slim
# # add your stuff here
# Now complete it:
# Install all the dependencies form the Pipenv file
# Copy your Flask script
# Run it with Gunicorn
# After that, you can build your docker image.


# Question 6
# Let's run your docker container!
# After running it, score this client once again:
# url = "YOUR_URL"
# client = {"job": "retired", "duration": 445, "poutcome": "success"}
# requests.post(url, json=client).json()
# What's the probability that this client will get a credit now?
# 0.168
# 0.530
# 0.730
# 0.968

# docker build -t mlhomework5 .
# docker run -it --rm -p 9696:9696 mlhomework5

# python predictdocker.py
output = {'poutcome': True, 'poutcome_probability': 0.726936946355423}

