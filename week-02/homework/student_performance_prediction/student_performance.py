# Imports and Settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(category=FutureWarning, action="ignore")
pd.set_option('display.max_columns', 0)

# Data file was filled with semicolon instead of comma, we have to fix this.
with open('week-2/homework/student_performance_prediction/student.csv', 'r') as file:
    data = file.read()

data = data.replace(';', ',')

with open('week-2/homework/student_performance_prediction/fixed_student.csv', 'w') as file:
    file.write(data)

# Basic EDA
df = pd.read_csv("week-2/homework/student_performance_prediction/fixed_student.csv")
df.head()
df.columns
df.isnull().sum()
df.dtypes

sns.histplot(df["G3"], color="red", bins=10, alpha=0.5)
sns.histplot(df["G3"], color="blue", bins=50, alpha=0.5)

# We have to make all variables numerical or we can work with only numerical values
categoricals = list(df.dtypes[df.dtypes == "object"].index)

np.random.seed(42)

# Splitting and shuffling the data set to df_train, df_test, df_val
n = len(df)
n_val = int(n * 0.20)
n_test = int(n * 0.20)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.shuffle(idx)
df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()

len(df_train), len(df_test), len(df_val)

y_train = np.log1p(df_train.G3.values)
y_test = np.log1p(df_test.G3.values)
y_val = np.log1p(df_val.G3.values)

del df_train["G3"]
del df_test["G3"]
del df_val["G3"]

# Scoring function RMSE(root mean squared error)
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


categorical = {}
for c in categoricals:
    categorical[c] = list(df_train[c].value_counts().head().index)

# Preparing the data set for predicting
def prepare_X(df):
    df = df.copy()
    features = []  # Initialize an empty list to store all feature names

    # Implementing one hot encoding
    for c in categoricals:
        dummies = pd.get_dummies(df[c], prefix=c)
        features.extend(dummies.columns.tolist())  # Add the dummy columns to the features list
        df = pd.concat([df, dummies], axis=1)  # Concatenate the dummy columns to the DataFrame

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

# Regularization
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]

# Tuning the model
for r in range(10, 75, 1):
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    print(r , w0, score)

# best 43 0.21925688529119866 0.8791299994940273

# Predicting with the model
X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X=X_train, y=y_train, r=43)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

sns.histplot(y_pred, color="red", bins=10, alpha=0.5)
sns.histplot(y_train, color="blue", bins=10, alpha=0.5)

score = rmse(y_val, y_pred)
print(score) # 0.8791299994940273
