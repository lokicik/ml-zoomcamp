import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(category=FutureWarning, action="ignore")
pd.set_option('display.max_columns', 0)

# Data Preparation
df = pd.read_csv("week-2/data.csv")
df.head()
df.columns = df.columns.str.lower().str.replace(" ", "_")

df.dtypes
strings = list(df.dtypes[df.dtypes == "object"].index)
strings
for col in strings:
    df[col] = df[col].str.lower().str.replace(" ", "_")

# Exploratory Data Analysis
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())

sns.histplot(df.msrp, bins=10)

sns.histplot(df.msrp, bins=50)

sns.histplot(df.msrp[df.msrp < 100000], bins=50)

np.log1p([0,1,10,100,99906.49631404002])
np.log([0+1,1+1,10+1,100+1,100000+1])

price_logs = np.log1p(df.msrp)
sns.histplot(price_logs, bins=50)

df.isnull().sum()

# Setting Up The Validation Framework
np.random.seed(2)
n = len(df)

n_val = int(n * 0.20)
n_test = int(n * 0.20)
n_train = int(n * 0.60)
n, n_val + n_train + n_test # (11914, 11912) !=

n_val = int(n * 0.20)
n_test = int(n * 0.20)
n_train = n - n_val - n_test
n, n_val + n_train + n_test # (11914, 11914) ==

idx = np.arange(n)


np.random.shuffle(idx)
df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()

len(df_train), len(df_test), len(df_val)

y_train = np.log1p(df_train.msrp.values)
y_test = np.log1p(df_test.msrp.values)
y_val = np.log1p(df_val.msrp.values)

del df_train["msrp"]
del df_test["msrp"]
del df_val["msrp"]

# Linear Regression
df_train.iloc[10]
xi = [155, 29, 586]
w0 = 7.17
w = [0.01, 0.04, 0.002]

def linear_regression(xi):
    n = len(xi)

    pred = w0

    for j in range(n):
        pred = pred + w[j] * xi[j]

    return pred

linear_regression(xi)
np.expm1(11.052000000000001)

# Linear Regression Vector Form
def dot(xi, w):
    n = len(xi)

    res = 0.0

    for j in range(n):
        res = res + xi[j] * w[j]
    return res

def linear_regression(xi):
    return w0 + dot(xi, w)

w_new = [w0] + w
w_new

def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w_new)

linear_regression(xi)

w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w

x1 = [1, 155, 29, 586]
x2 = [1, 148, 24, 1385]
x10 = [1, 132, 25, 2031]

X = [x1, x2, x10]
X = np.array(X)
X
def linear_regression(X):
    return X.dot(w_new)

linear_regression(X)

# Training a linear regression model
X=[
    [155, 29, 586],
    [148, 24, 1385],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38, 54, 185],
    [142, 25, 431],
    [453, 31, 86],
]
X = np.array(X)
ones = np.ones(X.shape[0])
X = np.column_stack([ones,X])

y = np.array([10000, 20000, 15000, 25000, 10000, 20000, 15000, 25000, 12000])

XTX = X.T.dot(X)

XTX_inv = np.linalg.inv(XTX)

w_full = XTX_inv.dot(X.T).dot(y)

w0 = w_full[0]
w = w_full[1:]

w0, w
np.de
X, y
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]



# Car price baseline model
df_train.columns

base = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg", "popularity"]
X_train = df_train[base].values
df_train[base].isnull().sum()
X_train = df_train[base].fillna(0).values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)

sns.histplot(y_pred, color="red", bins=50, alpha=0.5)
sns.histplot(y_train, color="blue", bins=50, alpha=0.5)

# RMSE
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

rmse(y_train, y_pred)

# Validating the model
base = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg", "popularity"]
X_train = df_train[base].values
df_train[base].isnull().sum()
X_train = df_train[base].fillna(0).values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)

def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)

# Simple feature engineering

def prepare_X(df):
    df = df.copy()

    df["age"] = 2017 - df.year
    features = base + ["age"]

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X

X_train = prepare_X(df_train)

w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)

sns.histplot(y_pred, color="red", bins=50, alpha=0.5)
sns.histplot(y_val, color="blue", bins=50, alpha=0.5)

# Categorical variables
df_train.number_of_doors
df.columns

def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
X_train = prepare_X(df_train)

w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
sns.histplot(y_pred, color="red", bins=50, alpha=0.5)
sns.histplot(y_val, color="blue", bins=50, alpha=0.5)


categorical_columns = ["engine_fuel_type","transmission_type","driven_wheels","market_category","vehicle_size","vehicle_style","make"]
categorical = {}
for c in categorical_columns:
    categorical[c] = list(df_train[c].value_counts().head().index)

def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for c, values in categorical.items():
        for v in values:
            df['%s_%s' % (c, v)] = (df[c] == v).astype("int")
            features.append("%s_%s" % (c, v))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


X_train = prepare_X(df_train)

w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)




# Regularization

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]
X_train = prepare_X(df_train)

w0, w = train_linear_regression_reg(X_train, y_train, r=0.0001)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)

# Tuning the model
for r in [0.0, 0.00001, 0.001, 0.1, 1, 10]:
    X_train = prepare_X(df_train)

    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    print(r , w0, score)


# Using the model
df_full_train = pd.concat([df_train, df_val])

df_full_train = df_full_train.reset_index(drop=True)

X_full_train = prepare_X(df_full_train)

X_full_train

y_full_train = np.concatenate([y_train, y_val])

w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.0001)

w

X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
score

car = df_test.iloc[20].to_dict()
df_small = pd.DataFrame([car])

X_small = prepare_X(df_small)

y_pred = w0 + X_small.dot(w)
y_pred = y_pred[0]
np.expm1(y_pred) # 10.4626591 / 34983.45487290064

np.expm1(y_test[20]) # 10.463131911491967 / 35000

