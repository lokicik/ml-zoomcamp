FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv flask waitress scikit-learn

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "predicthomework.py",  "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predicthomework:app"]