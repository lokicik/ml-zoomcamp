# # Question 5
# Download the base image agrigorev/zoomcamp-bees-wasps:v2.
# You can easily make it by using docker pull command.
# So what's the size of this base image?

# 162 Mb
# 362 Mb
# 662 Mb
# 962 Mb

# You can get this information when running docker images - it'll be in the "SIZE" column.


# docker pull agrigorev/zoomcamp-bees-wasps:v2
# docker images
# 662 MB


# Question 6
# Now let's extend this docker image,
# install all the required libraries and add the code for lambda.
# You don't need to include the model in the image.
# It's already included.
# The name of the file with the model
# is bees-wasps-v2.tflite and it's in the current workdir
# in the image (see the Dockerfile above for the reference).
# Now run the container locally.
# Score this image: https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg

# What's the output from the model?

# 0.2453
# 0.4453
# 0.6453
# 0.8453

# docker build -t homework9 .
# docker run -it --rm -p 8080:8080 homework9
# run test.py