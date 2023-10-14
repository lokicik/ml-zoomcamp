FROM python:3.8.12-slim

RUN pip install pipenv flask waitress scikit-learn

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "predict2.py", "model_C=1.0.bin","./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict2:app"]
