import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore")

# Data Preparation
df = pd.read_csv("week-3/homework/data.csv")
df.columns

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

df = df[needed_cols]
df.columns = df.columns.str.replace(' ', '_').str.lower()
df.rename(columns={"msrp":"price"}, inplace=True)
df.fillna(0, inplace=True)
df.isnull().any()

# Question 1
# What is the most frequent observation (mode) for the column transmission_type?
df["transmission_type"].mode()


# Question 2
# Create the correlation matrix for the numerical features of your dataset. In a correlation matrix, you compute the correlation coefficient between every pair of features in the dataset.
# What are the two features that have the biggest correlation in this dataset?
df.dtypes
numerical = ["year", "engine_hp", "engine_cylinders", "highway_mpg", "city_mpg"]
categorical = ["make", "model", "transmission_type", "vehicle_style"]
df[numerical].corrwith(df.price).sort_values(ascending=False)




# Make price binary
# Now we need to turn the price variable from numeric into a binary format.
# Let's create a variable above_average which is 1 if the price is above its mean value and 0 otherwise.
mean_price = df.price.mean()
df['above_average'] = (df['price'] > mean_price).astype(int)
df.above_average

# Split the data
# Split your data in train/val/test sets with 60%/20%/20% distribution.
# Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.
# Make sure that the target value (above_average) is not in your dataframe.
df_full_train, df_test = train_test_split(df, random_state=42, test_size=0.2)

df_train, df_val = train_test_split(df_full_train, random_state=42, test_size=0.25)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values

del df_train["above_average"]
del df_val["above_average"]
del df_test["above_average"]
# Question 3
# Calculate the mutual information score between above_average and other categorical variables in our dataset. Use the training set only.
# Round the scores to 2 decimals using round(score, 2).
# Which of these variables has the lowest mutual information score?

def mutual_info_above_average_score(series):
    return mutual_info_score(series, df_full_train.above_average)
mi = df_full_train[categorical].apply(mutual_info_above_average_score).sort_values(ascending=False).round(2)


# Question 4
# Now let's train a logistic regression.
# Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
# Fit the model on the training dataset.
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
# model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
# Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
# What accuracy did you get?
from sklearn.feature_extraction import DictVectorizer
train_dicts = df_train[categorical + numerical].to_dict(orient="records")
dv = DictVectorizer(sparse=False)
dv.fit(train_dicts)
dv.get_feature_names_out()
dv.transform(train_dicts)

X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
model.coef_
model.intercept_
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean().round(2) # 0.95


# Question 5
# Let's find the least useful feature using the feature elimination technique.
# Train a model with all these features (using the same parameters as in Q4).
# Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
# For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
# Which of following feature has the smallest difference?

df = pd.read_csv("week-3/homework/data.csv")
df.columns

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

df = df[needed_cols]
df.columns = df.columns.str.replace(' ', '_').str.lower()
df.rename(columns={"msrp":"price"}, inplace=True)
df.fillna(0, inplace=True)
df.isnull().any()

mean_price = df.price.mean()
df['above_average'] = (df['price'] > mean_price).astype(int)
df.above_average

df_full_train, df_test = train_test_split(df, random_state=42, test_size=0.2)

df_train, df_val = train_test_split(df_full_train, random_state=42, test_size=0.25)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values

del df_train["above_average"]
del df_val["above_average"]
del df_test["above_average"]


df_val.columns

results = pd.DataFrame(columns=["Feature", "Accuracy"])


features_to_analyze = ["make", "model", "year", "engine_hp", "engine_cylinders", "transmission_type", "vehicle_style",
                       "highway_mpg", "city_mpg", "price"]

numerical = ["year", "engine_hp", "engine_cylinders", "highway_mpg", "city_mpg"]
categorical = ["make", "model", "transmission_type", "vehicle_style"]

feature_differences = {}
all_list = categorical + numerical
i = 0
for feature in features_to_analyze:
    all_list.pop(i)
    train_dicts = df_train[all_list].to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dicts)

    dv.transform(train_dicts)

    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[all_list].to_dict(orient="records")
    X_val = dv.transform(val_dicts)



    model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)


    accuracy_filtered = accuracy_score(y_val, model.predict(X_val))


    feature_differences[feature] = accuracy_filtered
    all_list.insert(i, feature)
    i += 1



least_useful_feature = max(feature_differences, key=feature_differences.get)
smallest_difference = feature_differences[least_useful_feature]

print("Least useful feature:", least_useful_feature)
print("Smallest difference:", smallest_difference)
# 'city_mpg': 0.9324381032312211} smallest
# 'year': 0.9395719681074276,
# 'engine_hp': 0.9349559378934117,
# 'transmission_type': 0.9349559378934117,

# Question 6
# For this question, we'll see how to use a linear regression model from Scikit-Learn.
# We'll need to use the original column price. Apply the logarithmic transformation to this column.
# Fit the Ridge regression model on the training data with a solver 'sag'. Set the seed to 42.
# This model also has a parameter alpha. Let's try the following values: [0, 0.01, 0.1, 1, 10].
# Round your RMSE scores to 3 decimal digits.
# Which of these alphas leads to the best RMSE on the validation set?
# Apply logarithmic transformation to the 'price' column

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt
df['price_log'] = np.log1p(df['price'])


X_train, X_val = train_test_split(df, random_state=42, test_size=0.2)
y_train = X_train['price_log']
y_val = X_val['price_log']


alphas = [0, 0.01, 0.1, 1, 10]


best_rmse = float('inf')
best_alpha = None


for alpha in alphas:
    model = Ridge(alpha=alpha, solver='sag', random_state=42)
    model.fit(X_train[numerical], y_train)
    y_pred = model.predict(X_val[numerical])
    rmse = sqrt(mean_squared_error(y_val, y_pred))


    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha


best_rmse = round(best_rmse, 3)


print(f"The best alpha for Ridge regression is {best_alpha} with RMSE of {best_rmse}")





