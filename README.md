# DPP-Cache

## Introduction
This reposiroty contains datasets and files relevant to **DPP-caching algorithm**. We have also made use of GitHub Repository of **LeadCache** Algorithm for experimental comaprisions Links are given below.

- ***Kaggle***: https://www.kaggle.com/datasets/yoghurtpatil/311-service-requests-pitt
- ***LeadCache***: https://github.com/AbhishekMITIITM/LeadCache-NeurIPS21

## How to run
To run our algorithm follow the below steps:

1. Install python dependencies.
```
pip install -r requirements.txt
```
2. Change environemt variables in the *.env* file. A sample is shown below.
```
Q_INIT = 0                                # Inital Q value.
PAST = 3                                  # Number of previous slots used to predict
V_0 = 500                                 # Coeffecient of O(sqrt(T))
FUTURE = 1                                # Number of future slots to predict
ALPHA = 0.1                               # Percentage of catalogue as cache
NUM_SEQ = 300                             # Number of sequences
THRESHOLD = 423                           # Number of files in the catalogue
TRAIN_MEMORY = 5                          # Previous slots used to train
USE_SAVED = False                         # Whether to use saved model
RUN_OTHERS = True                         # Whether to run other algorithms
COST_CONSTRAINT = 20                      # Fetching cost Constraint
TIME_LIMIT = inf                          # Maximum requests per slot
PATH_TO_INPUT = Datasets/311_dataset.txt  # Path to request dataset
```

***Note:*** **Keep FUTURE key to be always 1**

3. Run the following command
```
python run.py
```
