import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import random

needed_cols = ["Make",
"Model",
"Year",
"Engine HP",
"Engine Cylinders",
"Transmission Type",
"Vehicle Style",
"highway MPG",
"city mpg",
"MSRP"]

# Data Preparation
# Keep only the columns above
# Lowercase the column names and replace spaces with underscores
# Fill the missing values with 0
# Make the price binary (1 if above the average, 0 otherwise) - this will be our target variable above_average
# Split the data into 3 parts: train/validation/test with 60%/20%/20% distribution. Use train_test_split function for that with random_state=1

df = pd.read_csv("week-4/homework/data.csv")
df = df[needed_cols]
df.columns = df.columns.str.replace(' ', '_').str.lower()
df.rename(columns={"msrp":"price"}, inplace=True)
df.fillna(0, inplace=True)
df.isnull().any()

numerical = ["year", "engine_hp", "engine_cylinders", "highway_mpg", "city_mpg"]
categorical = ["make", "model", "transmission_type", "vehicle_style"]
mean_price = df.price.mean()
df['above_average'] = (df['price'] > mean_price).astype(int)
df.above_average
df.drop("price", axis=1, inplace=True)
df.columns
df_full_train, df_test = train_test_split(df, random_state=1, test_size=0.2)

df_train, df_val = train_test_split(df_full_train, random_state=1, test_size=0.25)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values

del df_train["above_average"]
del df_val["above_average"]
del df_test["above_average"]

# Question 1: ROC AUC feature importance
# ROC AUC could also be used to evaluate feature importance of numerical variables.
# For each numerical variable, use it as score and compute AUC with the above_average variable
# Use the training dataset for that
# If your AUC is < 0.5, invert this variable by putting "-" in front
# (e.g. -df_train['engine_hp'])
# AUC can go below 0.5 if the variable is negatively correlated with the target varialble. You can change the direction of the correlation by negating this variable - then negative correlation becomes positive.
# Which numerical variable (among the following 4) has the highest AUC?

numericals_to_select = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg"]

iterate_auc = 0
iterate_variable = None

for variable in numericals_to_select:
    auc = roc_auc_score(y_train, df_train[variable])

    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[variable])

    iterate_auc = auc
    iterate_variable = variable
    print("\nNumerical variable :", iterate_variable)
    print("AUC score:", iterate_auc)

# Numerical variable : engine_hp
# AUC score: 0.9171031265539011

# Question 2: Training the model
# Apply one-hot-encoding using DictVectorizer and train the logistic regression with these parameters:
# LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
# What's the AUC of this model on the validation dataset? (round to 3 digits)

# first try
dv = DictVectorizer(sparse=True)
train_dicts = df_train[categorical + numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]  # Probability of positive class
score = roc_auc_score(y_val, y_pred).round(3)
print("AUC on the validation dataset:", score)


# second try
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]
import random
n = 100000
success = 0

for i in range(n):
    np.random.seed(1)
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

np.round(success / n, 3)

# third try
n = 50000
np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)
(pos[pos_ind] > neg[neg_ind]).mean().round(3)

# 0.98, nearest option is 0.979

# Question 3: Precision and Recall
# Now let's compute precision and recall for our model.
# Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01
# For each threshold, compute precision and recall
# Plot them
# At which threshold precision and recall curves intersect?
thresholds = np.linspace(0, 1, 101)
precisions = []
recalls = []

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    tp = (predict_positive & actual_positive).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions, label="Precision", linestyle='--')
plt.plot(thresholds, recalls, label="Recall", linestyle='-')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision and Recall Curves")
plt.grid(True)


custom_y_ticks = [i/20 for i in range(21)]
plt.yticks(custom_y_ticks)

plt.show()
# The intersection is near 0.46-0.50, nearest option is 0.48

# Question 4: F1 score
# Precision and recall are conflicting - when one grows, the other goes down. That's why they are often combined into the F1 score - a metrics that takes into account both
# This is the formula for computing F1:
# F1 = 2(P.R)/(P+R)
# Where is P precision and R is recall.
# Let's compute F1 for all thresholds from 0.0 to 1.0 with increment 0.01
# At which threshold F1 is maximal?
thresholds = np.linspace(0, .99, 100)
scores = {}

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    p = tp / (tp + fp)

    r = tp / (tp + fn)
    score = 2 * (p * r) / (p + r)

    scores[t] = score


scores
value_to_find = max(list(scores.values()))
keys_with_value = [key for key, value in scores.items() if value == value_to_find]

# value_to_find
# Out[203]: 0.8907563025210082
# keys_with_value
# Out[204]: [0.49]
# The nearest option is 0.52

# Question 5: 5-Fold CV
# Use the KFold class from Scikit-Learn to evaluate our model on 5 different folds:
# KFold(n_splits=5, shuffle=True, random_state=1)
# Iterate over different folds of df_full_train
# Split the data into train and validation
# Train the model on train with these parameters: LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
# Use AUC to evaluate the model on validation
# How large is standard deviation of the scores across different folds?

kf = KFold(n_splits=5, shuffle=True, random_state=2)

auc_scores = []

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)

dv = DictVectorizer(sparse=False)

for train_idx, val_idx in kf.split(df_full_train):

    df_train_fold = df_full_train.iloc[train_idx]
    df_val_fold = df_full_train.iloc[val_idx]

    y_train_fold = df_train_fold.above_average.values
    y_val_fold = df_val_fold.above_average.values


    del df_train_fold["above_average"]
    del df_val_fold["above_average"]

    train_dicts = df_train_fold.to_dict(orient='records')
    val_dicts = df_val_fold.to_dict(orient='records')

    X_train_fold = dv.fit_transform(train_dicts)
    X_val_fold = dv.transform(val_dicts)

    model.fit(X_train_fold, y_train_fold)

    y_pred_fold = model.predict_proba(X_val_fold)[:, 1]
    auc_fold = roc_auc_score(y_val_fold, y_pred_fold)
    auc_scores.append(auc_fold)

print("Standard Deviation of AUC Scores:", np.std(auc_scores))
# Standard Deviation of AUC Scores: 0.0027216520271203474

# Question 6: Hyperparameter Tuning
# Now let's use 5-Fold cross-validation to find the best parameter C
# Iterate over the following C values: [0.01, 0.1, 0.5, 10]
# Initialize KFold with the same parameters as previously
# Use these parameters for the model: LogisticRegression(solver='liblinear', C=C, max_iter=1000)
# Compute the mean score as well as the std (round the mean and std to 3 decimal digits)
# Which C leads to the best mean score?
# If you have ties, select the score with the lowest std. If you still have ties, select the smallest C.

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

dv, model = train(df_train, y_train, C=0.001)

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

y_pred = predict(df_val, dv, model)

from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

len(y_val), len(y_pred)
n_splits = 5

for C in [0.01, 0.1, 0.5, 10]:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.above_average.values
        y_val = df_val.above_average.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('\nC=%s %.3f +- %.3f\n' % (C, np.mean(scores), np.std(scores)))


# BEST FOR C=10 0.980 +- 0.003






