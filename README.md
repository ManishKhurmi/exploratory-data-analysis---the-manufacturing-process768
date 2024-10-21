# exploratory-data-analysis---the-manufacturing-process768

## Installation instructions
Run in the following order:
1. manufacturing_EDA.ipynb
2. 


## Table of Contents, if the README file is long

## A description of the project: what it does, the aim of the project, and what you learned

This project focuses on optimising a key manufacturing process for a large industrial company, aiming to improve efficiency and profitability by reducing waste and maximising output. Using logistic regression to predict machine failure and analysing the local minima of the second derivative of machine performance, I derived operational thresholds to prevent failures and enhance performance.

The dataset contains information from 10,000 production sessions, including machine states and failure reasons. By identifying optimal operating thresholds, the analysis offers a clear understanding of when and why the machinery is likely to fail, allowing the company to take proactive steps to reduce the failure rate and increase productivity.

 What I Learned:

**Project Phases Matter**: I found that aiming to stick with the process—Task definition, Analysis, Development, and Testing—made the project run much smoother. Spending more time on defining the task and analysing what needs to be done upfront saved me from reworking during development and helped me move forward with clarity.

**Data Types Guide Model Selection**: Understanding the data types early on made model selection easier. Since I had independent variables that were numeric and binary, I knew that logistic regression would be a good fit for this classification task. Logistic regression is ideal for binary outcomes because it models the probability of an event occurring (such as machine failure) and outputs values between 0 and 1. This made it an obvious choice for predicting machine failure.

**Combining Statistics and Visuals**: I saw firsthand how powerful it can be to combine statistical modelling with clear visualisations. This combination allowed me to turn complex data into actionable insights that could directly support business decisions.

## Usage instructions

### Option 1: Using Conda (Basic Setup - No Database Integration)

If you only want to see the EDA of the manufacturing data set and do not want to extract the data yourself from the cloud (i.e. run the `db_utils.py`), you can create the Conda Environment with the following command:

```bash
conda env create -f environment.yml
```
### Option 2: Using Conda (Full Setup - Includes Database Integration)

```bash
conda env create -f environment_with_db.yml
```
Tou may need the following system level packages:

```
brew install libpq
brew link --force libpq
```
### Option 3: Using pip install for your own virtual env

```
pip install PyYAML sqlalchemy psycopg2 pandas numpy matplotlib seaborn scipy statsmodels ipython scikit-learn
```

## File structure of the project
![alt text](<Screenshot 2024-10-19 at 09.04.24.png>)

## License information
Open to all 


