import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(category=FutureWarning, action="ignore")
pd.set_option('display.max_columns', 0)

main_df = pd.read_csv("week-2/homework/housing.csv")
df = main_df[(main_df['ocean_proximity'] == '<1H OCEAN') | (main_df['ocean_proximity'] == 'INLAND')]
df["median_house_value"].tail()
needed_cols = ['latitude',
    'longitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'median_house_value']

for col in df.columns:
    if col not in needed_cols:
        df.drop(col, axis=1, inplace=True)
df.columns
# Question 1
# There's one feature with missing values. What is it?
df.isnull().any() # total_bedrooms         True

# Question 2
# What's the median (50% percentile) for variable 'population'?
df["population"].median() # 1195.0

# Prepare and split the dataset
# Shuffle the dataset (the filtered one you created above), use seed 42.
# Split your data in train/val/test sets, with 60%/20%/20% distribution.
# Apply the log transformation to the median_house_value variable using the np.log1p() function.
def load_dataset():
    main_df = pd.read_csv("week-2/homework/housing.csv")
    return main_df

def prepare_set(main_df, seed=42):
    df = main_df[(main_df['ocean_proximity'] == '<1H OCEAN') | (main_df['ocean_proximity'] == 'INLAND')]
    df["median_house_value"].tail()
    needed_cols = ['latitude',
                   'longitude',
                   'housing_median_age',
                   'total_rooms',
                   'total_bedrooms',
                   'population',
                   'households',
                   'median_income',
                   'median_house_value']

    for col in df.columns:
        if col not in needed_cols:
            df.drop(col, axis=1, inplace=True)
    np.random.seed(seed)
    n = len(df)

    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    idx = np.arange(n)

    np.random.shuffle(idx)
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    y_train = np.log1p(df_train.median_house_value.values)
    y_val = np.log1p(df_val.median_house_value.values)
    y_test = np.log1p(df_test.median_house_value.values)

    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']
    return df_train, df_val, df_test, y_train, y_val, y_test


# Question 3
# We need to deal with missing values for the column from Q1.
# We have two options: fill it with 0 or with the mean of this variable.
# Try both options. For each, train a linear regression model without regularization using the code from the lessons.
# For computing the mean, use the training only!
# Use the validation dataset to evaluate the models and compare the RMSE of each option.
# Round the RMSE scores to 2 decimal digits using round(score, 2)
# Which option gives better RMSE?
# Options:
# With 0
# With mean
# Both are equally good
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

# Fill with 0 RMSE
main_df = load_dataset()

df_train, df_val, df_test, y_train, y_val, y_test = prepare_set(main_df)

def prepare_X(df):
    df = df.copy()
    df_num = df.fillna(0)
    X = df_num.values
    return X

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]



X_train = prepare_X(df_train)

w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

with_zero = rmse(y_val, y_pred)
round(with_zero, 2) # 0.34



# Fill with mean RMSE : 0.33797453660498306
df = load_dataset()
median_to_fill = df_train["total_bedrooms"].mean()
df_train, df_val, df_test, y_train, y_val, y_test = prepare_set(df)

def prepare_X(df):
    df = df.copy()
    df_num = df.fillna(median_to_fill)
    X = df_num.values
    return X


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
with_mean = rmse(y_val, y_pred)
round(with_mean, 2) # 0.34

# Question 4
# Now let's train a regularized linear regression.
# For this question, fill the NAs with 0.
# Try different values of r from this list: [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10].
# Use RMSE to evaluate the model on the validation dataset.
# Round the RMSE scores to 2 decimal digits.
# Which r gives the best RMSE?
# If there are multiple options, select the smallest r.
# Options:
# 0
# 0.000001
# 0.001
# 0.0001
main_df = load_dataset()

df_train, df_val, df_test, y_train, y_val, y_test = prepare_set(main_df)

def prepare_X(df):
    df = df.copy()
    df_num = df.fillna(0)
    X = df_num.values
    return X

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]


# Tuning the model
for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    X_train = prepare_X(df_train)

    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    print(r , w0, score)

X_train = prepare_X(df_train)

w0, w = train_linear_regression_reg(X_train, y_train, r=0.0)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

with_zero = rmse(y_val, y_pred)
round(with_zero, 2) # 0.34



# Question 5
# We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
# Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
# For each seed, do the train/validation/test split with 60%/20%/20% distribution.
# Fill the missing values with 0 and train a model without regularization.
# For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
# What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
# Round the result to 3 decimal digits (round(std, 3))
# What's the value of std?
# 0.5
# 0.05
# 0.005
# 0.0005
scores = []
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    main_df = load_dataset()
    df_train, df_val, df_test, y_train, y_val, y_test = prepare_set(main_df, seed=seed)

    def prepare_X(df):
        df = df.copy()
        df_num = df.fillna(0)
        X = df_num.values
        return X


    def train_linear_regression(X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)

        return w[0], w[1:]


    X_train = prepare_X(df_train)
    w0, w = train_linear_regression(X_train, y_train)
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    scores.append(score)

scores
round(np.std(a=scores), 3) # 0.005

# Question 6
# Split the dataset like previously, use seed 9.
# Combine train and validation datasets.
# Fill the missing values with 0 and train a model with r=0.001.
# What's the RMSE on the test dataset?
# Options:
# 0.13
# 0.23
# 0.33
# 0.43
main_df = load_dataset()
df_train, df_val, df_test, y_train, y_val, y_test = prepare_set(main_df, seed=9)

def prepare_X(df):
    df = df.copy()
    df_num = df.fillna(0)
    X = df_num.values
    return X


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]


df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
X_full_train = prepare_X(df_full_train)
y_full_train = np.concatenate([y_train, y_val])
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
rmse(y_test, y_pred) # 0.33498993366084406
