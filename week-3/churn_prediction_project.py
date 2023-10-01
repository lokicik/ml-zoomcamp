import pandas as pd
import numpy as np
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings(action="ignore")

# Data Preparation

df = pd.read_csv("week-3/Telco_Customer_Churn.csv")

df.head().T

df.columns = df.columns.str.lower().str.replace(" ", "_")
categoricals = list(df.dtypes[df.dtypes == "object"].index)

for c in categoricals:
    df[c] = df[c].str.lower().str.replace(" ", "_")

df.dtypes
df.totalcharges
tc = pd.to_numeric(df.totalcharges, errors="coerce")

df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == "yes").astype(int)


# Setting up the Validation Framework
from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=1)

len(df_train), len(df_val), len(df_test)

df
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train["churn"]
del df_val["churn"]
del df_test["churn"]

# EDA
df_full_train = df_full_train.reset_index(drop=True)
df_full_train.isnull().sum()
df_full_train.churn.value_counts(normalize=True)
df_full_train.churn.mean()

global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate, 2)

df_full_train.dtypes

numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

df_full_train[categorical].nunique()

# Feature Importance: Churn rate and risk ratio

# churn rate
df_full_train.head()
churn_female = df_full_train[df_full_train.gender == "female"].churn.mean()
churn_male = df_full_train[df_full_train.gender == "male"].churn.mean()
global_churn_rate

churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
global_churn_rate - churn_partner

churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
global_churn_rate - churn_no_partner

# risk ratio
churn_no_partner / global_churn_rate
churn_partner / global_churn_rate

for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(["mean", "count"])
    df_group["diff"] = df_group["mean"] - global_churn_rate
    df_group["risk"] = df_group["mean"] / global_churn_rate
    print(c+"\n",df_group,"\n\n")

# Feature importance: Mutual information
from sklearn.metrics import mutual_info_score
mutual_info_score(df_full_train.churn, df_full_train.contract)
mutual_info_score(df_full_train.churn, df_full_train.gender)

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)
mi = df_full_train[categorical].apply(mutual_info_churn_score).sort_values(ascending=False)


# Feature importance: Correlation
df_full_train[numerical].corrwith(df_full_train.churn).abs()

df_full_train[df_full_train.tenure <= 2].churn.mean()
df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()
df_full_train[df_full_train.tenure > 12].churn.mean()


# One-Hot Encoding
from sklearn.feature_extraction import DictVectorizer
train_dicts = df_train[categorical + numerical].to_dict(orient="records")
dv = DictVectorizer(sparse=False)
dv.fit(train_dicts)
dv.get_feature_names_out()
dv.transform(train_dicts)

X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)

dv = DictVectorizer(sparse=False)



# Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-7, 7, 51)
sigmoid(z)

plt.plot(z, sigmoid(z))

# def linear_regression(xi):
#     result = w0
#     for j in range(len(w)):
#         score = score + xi[j] * w[j]
#     return result

# def logistic_regression(xi):
#     score = w0
#     for j in range(len(w)):
#         score = score + xi[j] * w[j]
#     result = sigmoid(score)
#     return result

# Training logistic regression with scikit-learn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.coef_
model.intercept_
# model.predict(X_train) # hard predictions
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
df_val[churn_decision].customerid
(y_val == churn_decision).mean()

df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val

df_pred['correct'] = df_pred.prediction == df_pred.actual
df_pred

# model interpretation
a = [1, 2, 3, 4]
b = 'abcd'
dict(zip(a, b))
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))


small = ['contract', 'tenure', 'monthlycharges']
df_train[small].iloc[:10].to_dict(orient='records')

dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')

dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)
dv_small.get_feature_names_out()
X_train_small = dv_small.transform(dicts_train_small)



model_small = LogisticRegression()
model_small.fit(X_train_small, y_train)
w0 = model_small.intercept_[0]
w0
w = model_small.coef_[0]
w.round(3)
dict(zip(dv_small.get_feature_names_out(), w.round(3)))
-2.47 + (-0.949) + 30 * 0.027 + 24 * (-0.036)
sigmoid(_)


# Using the model
dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

y_full_train = df_full_train.churn.values

model = LogisticRegression()
model.fit(X_full_train, y_full_train)

dicts_test = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(dicts_test)
y_pred = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred >= 0.5)
(churn_decision == y_test).mean()
y_test
customer = dicts_test[-1]
customer
X_small = dv.transform([customer])
model.predict_proba(X_small)[0, 1]
y_test[-1]