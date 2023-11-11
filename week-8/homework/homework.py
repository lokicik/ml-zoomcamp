# Data Prep


# Question 1
# Since we have a binary classification problem, what is the best loss function for us?
#
# mean squared error
# binary crossentropy
# categorical crossentropy
# cosine similarity
# Note: since we specify an activation for the output layer, we don't need to set from_logits=True


# Question 2
# What's the number of parameters in the convolutional layer of our model? You can use the summary method for that.
#
# 1
# 65
# 896
# 11214912


# Generators and Training
# For the next two questions, use the following data generator for both train and test sets:
#
# ImageDataGenerator(rescale=1./255)
# We don't need to do any additional pre-processing for the images.
# When reading the data from train/test directories, check the class_mode parameter. Which value should it be for a binary classification problem?
# Use batch_size=20
# Use shuffle=True for both training and test sets.
# For training use .fit() with the following params:
#
# model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=test_generator
# )


# Question 3
# What is the median of training accuracy for all the epochs for this model?
#
# 0.20
# 0.40
# 0.60
# 0.80


# Question 4
# What is the standard deviation of training loss for all the epochs for this model?
#
# 0.31
# 0.61
# 0.91
# 1.31

# Data Augmentation
# For the next two questions, we'll generate more data using data augmentations.
#
# Add the following augmentations to your training data generator:
#
# rotation_range=50,
# width_shift_range=0.1,
# height_shift_range=0.1,
# zoom_range=0.1,
# horizontal_flip=True,
# fill_mode='nearest'


# Question 5
# Let's train our model for 10 more epochs using the same code as previously.
#
# Note: make sure you don't re-create the model - we want to continue training the model we already started training.
#
# What is the mean of test loss for all the epochs for the model trained with augmentations?
#
# 0.18
# 0.48
# 0.78
# 0.108



# Question 6
# What's the average of test accuracy for the last 5 epochs (from 6 to 10) for the model trained with augmentations?
#
# 0.38
# 0.58
# 0.78
# 0.98
