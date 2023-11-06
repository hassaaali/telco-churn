# Churn Rate 1.0

Welcome to my machine learning project! Below, you'll find information about the how to run the program in Docker. 

The machine learning application predicts the churn of a customer i.e. predicts how likely it is that a customer will unscubscribe from a company's telecommunication services. The average accuracy of the machine learning application is 78% using the Random Forest ML model. The training data has been sourced from Kaggle (https://www.kaggle.com/datasets/reyhanarighy/data-telco-customer-churn).

## Software Packages

All the packages required to run the application are included in `requirements.txt`

## Docker Installation

To install Docker, you can follow the installation guide on the official Docker website:

[Install Docker](https://docs.docker.com/engine/install/)

This guide provides instructions for installing Docker on various platforms.

## Setting up a Docker Environment

To ensure a clean and isolated environment for your project, you can create a container using `docker`. Here are the steps to create a docker container:

1. Open your terminal and clone the repository.
2. Install `docker` if you haven't already.
3. Create a docker image of the machine learning environment named `ml_env`:

   ```docker build -t ml_env:v4 .```

4. Create and run a docker conatiner using the image created (The conatiner uses port 5000 for communicatio with GUI):

   ```docker run --rm -it -p 5000:5000/tcp ml_env```

## Running the frontend dashboard:

1. Change directory to `gui'
2. Install the requirements included in `requirements.txt`
3. Run the app:

   ```python3 main.py```

4. You can now access the app through http://localhost:8000/telco-churn/ and get the churn result for user defined inputs.

Note: The GUI dashboard can also run in a docker container instead (optional)

## Swagger:

Swagger Endpoint: http://localhost:5000/telco-churn-api/docs