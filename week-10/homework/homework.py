import requests
from time import sleep


url = "http://localhost:9696/predict"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}

while True:
    sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)


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
# kubectl


# Q4