import requests
from time import sleep


url = "http://localhost:9696/predict"


# Q1

# git clone https://github.com/DataTalksClub/machine-learning-zoomcamp.git
# docker build -t zoomcamp-model:hw10 .
# docker run -it --rm -p 9696:9696 zoomcamp-model:hw10
# python q6_test.py

#### {'get_credit': True, 'get_credit_probability': 0.726936946355423}


# Q2
# kind --version


# Creating a cluster

# kind create cluster
# kubectl cluster-info


# Q3
# kubectl get services


# Q4
# kind load docker-image zoomcamp-model:hw10

