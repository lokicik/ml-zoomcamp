import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from xgboost import DMatrix
from sklearn.tree import plot_tree
# Data Prep
df = pd.read_csv("week-6/homework/housing.csv")


df = df.fillna(0)
df = df[(df['ocean_proximity'] == '<1H OCEAN') | (df['ocean_proximity'] == 'INLAND')]

df["median_house_value"] = np.log1p(df.median_house_value)
# np.expm1()

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.median_house_value.values
y_val = df_val.median_house_value.values
y_test = df_test.median_house_value.values

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']

dv = DictVectorizer(sparse=True)

X_train = dv.fit_transform(df_train[df_train.columns].to_dict(orient='records'))
X_val = dv.transform(df_val[df_val.columns].to_dict(orient='records'))
X_test = dv.transform(df_test[df_test.columns].to_dict(orient='records'))


# Question 1
# Let's train a decision tree regressor to predict the median_house_value variable.
# Train a model with max_depth=1.
# Which feature is used for splitting the data?

model = DecisionTreeRegressor(max_depth=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")


# Visualize
plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=dv.get_feature_names_out(input_features=df.columns))
plt.show()
# ocean_proximity



# Question 2
# Train a random forest model with these parameters:

# n_estimators=10
# random_state=1
# n_jobs=-1 (optional - to make training faster)

# What's the RMSE of this model on validation?
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
rmse = (mean_squared_error(y_val, y_pred))**(1/2)
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
# Root Mean Squared Error (RMSE): 0.0245





# Question 3
# Now let's experiment with the n_estimators parameter
# Try different values of this parameter from 10 to 200 with step 10.
# Set random_state to 1.
# Evaluate the model on the validation dataset.
# After which value of n_estimators does RMSE stop improving?
for n_estimators in range(10,200,10):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = (mean_squared_error(y_val, y_pred)) ** (1 / 2)
    print(f"n_estimators:{n_estimators} | RMSE: {rmse:.3f}")

n_estimator_dict = {
 10:0.245,
 20:0.239,
 30:0.237,
 40:0.236,
 50:0.235,
 60:0.235,
 70:0.234,
 80:0.235,
 90:0.235,
 100:0.234,
 110:0.234,
 120:0.234,
 130:0.234,
 140:0.234,
 150:0.234,
 160:0.233,
 170:0.233,
 180:0.234,
 190:0.234
}

# step 160




# Question 4
# Let's select the best max_depth:
# Try different values of max_depth: [10, 15, 20, 25]
# For each of these values, try different values of n_estimators from 10 till 200 (with step 10)
# Fix the random seed: random_state=1
# What's the best max_depth?
for max_depth in [10, 15, 20, 25]:
    rf = RandomForestRegressor(max_depth, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = (mean_squared_error(y_val, y_pred)) ** (1 / 2)
    print(f"max_depth: {max_depth} | RMSE: {rmse:.3f}")

# max_depth: 10 | RMSE: 0.245
# max_depth: 15 | RMSE: 0.242
# max_depth: 20 | RMSE: 0.239
# max_depth: 25 | RMSE: 0.238 ***


# Question 5
# We can extract feature importance information from tree-based models.
# At each step of the decision tree learning algorithm, it finds the best split.
# When doing it, we can calculate "gain" - the reduction in impurity before and after the split.
# This gain is quite useful in understanding what are the important features for tree-based models.
# In Scikit-Learn, tree-based models contain this information in the feature_importances_ field.
# For this homework question, we'll find the most important feature:

# Train the model with these parameters:
# n_estimators=10,
# max_depth=20,
# random_state=1,
# n_jobs=-1 (optional)

# Get the feature importance information from this model
# What's the most important feature (among these 4)?
model = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

feature_names = df_train.columns
feature_importances = model.feature_importances_
feature_importance_dict = dict(zip(feature_names, feature_importances))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_feature_importance:
    print(f"Feature: {feature}, Importance: {importance:.4f}")

# Feature: total_bedrooms, Importance: 0.3356 ***
# Feature: population, Importance: 0.2925
# Feature: housing_median_age, Importance: 0.1020
# Feature: total_rooms, Importance: 0.0862
# Feature: households, Importance: 0.0738
# Feature: latitude, Importance: 0.0303
# Feature: median_income, Importance: 0.0271
# Feature: ocean_proximity, Importance: 0.0159
# Feature: longitude, Importance: 0.0151




# Question 6
# Now let's train an XGBoost model! For this question, we'll tune the eta parameter:
# Install XGBoost
# Create DMatrix for train and validation
# Create a watchlist
# Train a model with these parameters for 100 rounds:
# xgb_params = {
#     'eta': 0.3,
#     'max_depth': 6,
#     'min_child_weight': 1,
#     'objective': 'reg:squarederror',
#     'nthread': 8,
#     'seed': 1,
#     'verbosity': 1,
# }
# Now change eta from 0.3 to 0.1.
# Which eta leads to the best RMSE score on the validation dataset?



# Define the XGBoost parameters
xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# Create DMatrix for the training and validation sets
dtrain = DMatrix(X_train, label=y_train)
dval = DMatrix(X_val, label=y_val)

# Create a watchlist
watchlist = [(dtrain, 'train'), (dval, 'validation')]

# Train the model with eta = 0.3
num_round = 100
model_eta_0_3 = xgb.train(xgb_params, dtrain, num_round, evals=watchlist)

# Make predictions with the model
y_pred_eta_0_3 = model_eta_0_3.predict(dval)

# Calculate RMSE for eta = 0.3
rmse_eta_0_3 = mean_squared_error(y_val, y_pred_eta_0_3, squared=False)
print(f"RMSE (eta = 0.3): {rmse_eta_0_3:.5f}")
# RMSE (eta = 0.3): 0.22862


# Now change eta to 0.1
xgb_params['eta'] = 0.1

# Train the model with eta = 0.1
model_eta_0_1 = xgb.train(xgb_params, dtrain, num_round, evals=watchlist)

# Make predictions with the model
y_pred_eta_0_1 = model_eta_0_1.predict(dval)

# Calculate RMSE for eta = 0.1
rmse_eta_0_1 = mean_squared_error(y_val, y_pred_eta_0_1, squared=False)
print(f"RMSE (eta = 0.1): {rmse_eta_0_1:.5f}")
# RMSE (eta = 0.1): 0.23209