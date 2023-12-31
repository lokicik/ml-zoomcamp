# Evaluations Done

For hash: 774678aec551cc8dd1fbee0595bf39af48ea7322

Problem description: 0 point

Exploratory data analysis: 2 points

Model training: 2 points

Exporting notebook to script: 1 point

Reproducibility: 1 point

Dependency and environment management: 2 points

Containerization: 2 points

Cloud deployment: 2 points

total: 13 points

Notes: 
Could have done hyperparameter optimization a little bit to get 3 points from model training, and in the notebook, markdowns could be used more creatively. Problem description is not enough, there is a description for the data, deployment and environment management, but no sign of any problem statements, just an overview section which explains the aim of the project. Except these, I can easily reproduce the scripts and understand the code. I like that you explain the evaluation metrics in the README. Nice project! Keep developing more!

---------------------------------------------------

For hash: 278213ecf9dee70f7cecd7e3b394bc192c026f70

Problem description: 1 point

Exploratory data analysis: 1 points

Model training: 2 points

Exporting notebook to script: 0 point

Reproducibility: 0 point

Model deployment: 0 point

Dependency and environment management: 2 points

Containerization: 0 points

Cloud deployment: 0 points

total: 6 points

Notes: 
I was able to clone the repo, activated the pipenv according to the instructions and checked if the packages are loaded correctly, but I wasn't able to run the notebook.ipynb, I tried to fix the errors, but I failed. It is really hard to read your notebook by the way. EDA is simple, I see a lot of visualizations and some feature engineering. I understand that you created the data yourself, but it's not appropriate to not do missing values and outlier analysis, they are a part of EDA.  I see that you have trained multiple models but there was no hyperparameter optimizations. train.py was empty. There is some instructions about model deployment, but I see no such thing like "predict.py" in your files. There is a Dockerfile, but it's incomplete as I can tell. There is no cloud deployment. Your evaluation points really check out with your Self-Check, but please don't lose your courage! You can still finish this project, even if it won't be evaluated. I believe in you!

--------------------------------------------------

For hash: 61fe8e7dd94903436c0fd8fd23ef88a788674b77
Problem description: 0 point
Exploratory data analysis: 2 points
Model training: 2 points
Exporting notebook to script: 0 point
Reproducibility: 0 point
Model deployment: 1 point
Dependency and environment management: 2 points
Containerization: 2 points
Cloud deployment: 0 points
total: 9 points

Notes: 
I cloned the repo and loaded the pipenv without any problems. There is nothing like a problem statement or like that. I see that you trained multiple models, but no hyperparameter optimizations. For EDA there is missing value analysis and outlier detection and some visualizations with feature engineering, there could be a correlation matrix and feature importance analysis. There is no train.py script for me to run and reproduce the results. There is a script for web deployment with flask. There is a Dockerfile and info about containerization which seems to be working. Finally, there is no cloud deployment. I think you can do better, starting with your notebook order. Don't give up!

------------------------------------------------------------
# My Evaluations
1-)
In general I liked the project; however I did not actually get the justification, if we know feature variables for a patient, then we already know if his health is good or bad, there is no need to predict if he is a potential smoker, moreover I could not get how the model assesses the "the harm caused by passive smoking" or "it can be used to identify whether children have been exposed to smoking or alcohol" (your dataset includes people over 20, it did not see anybody younger, if it produces the meaningful results? who knows). The problem is that those features used for a model are most probably not the best predictors for smokers/alc, this clearly indicates your model with an accuracy 70% and recall 70%, you misclassify a lot and the reason is most probably, that from biological point of view, smoker/non smokers can have the same features (pressure, chol, etc, especially if smb started smoking a year ago) and those are not enough to state whether somebody is a smoker or nor. Your EDA indicates it clearly,the distribution of features stratified on smoker/non smoker is very similar for lots of predictors. Some heatingmaps are just too large to use them, perhaps it would be better to correlate your target variables with predictors instead and make a plot. A VotingClassifier is used in the project, however it seems that the separate models built for the problem are very similar, so why is it used? You should be also careful with throwing out outliers (as you define them, they are those outside a whisker box), disease state is an outlier itself, or at least we assume that smoking or drinking alc increase significantly some features!


2-)
great work, I learned a lot with your project. Congratulations!! I would just give you advice to add more comments throughout the EDA so that it is easier to understand your line of reasoning throughout the analysis.


3-)
Regarding reproducibility:
pipenv shell instead of pip shell

Your train script should only contain the logic for training. The notebook already entails the EDA, parameter tuning etc.