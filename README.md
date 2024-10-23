# Exploratory Data Analysis - The Manufacturing Process

## Table of Contents
1. [Installation Instructions](#installation-instructions)
   - [Option 1: Using Conda (Basic Setup - No Database Integration)](#option-1-using-conda-basic-setup---no-database-integration)
   - [Option 2: Using Conda (Full Setup - Includes Database Integration)](#option-2-using-conda-full-setup---includes-database-integration)
   - [Option 3: Using pip Install](#option-3-using-pip-install)
   - [Option 4: Using a Python Virtual Environment (Basic Setup - No Database Integration)](#option-4-using-a-python-virtual-environment-basic-setup---no-database-integration)
   - [Option 5: Using a Python Virtual Environment (Full Setup - Includes Database Integration)](#option-5-using-a-python-virtual-environment-full-setup---includes-database-integration)
2. [Project Description](#project-description)
3. [Usage Instructions](#usage-instructions)
4. [File Structure of the Project](#file-structure-of-the-project)
5. [Project Workflow](#project-workflow)
6. [License Information](#license-information)


## Installation instructions
### Option 1: Using Conda (Basic Setup - No Database Integration)

If you only want to see the EDA of the manufacturing data set and do not want to extract the data yourself from the cloud (i.e. run the `db_utils.py`), you can create the Conda Environment with the following command:

```bash
conda env create -f environment.yml
```
### Option 2: Using Conda (Full Setup - Includes Database Integration)

```bash
conda env create -f environment_with_db.yml
```
You may need the following system level packages too:

```
brew install libpq
brew link --force libpq
```
### Option 3: Using pip install 

```
pip install PyYAML sqlalchemy psycopg2 pandas numpy matplotlib seaborn scipy statsmodels ipython scikit-learn
```
### Option 4: Using a python virtual environement (Basic Setup - No Database Integration)

```
pip install -r requirements.txt
```

### Option 5: Using a python virtual environement (Full Setup - Includes Database Integration)

```
pip install -r requirements_with_db.txt
```

## Table of Contents, if the README file is long

## Project Description

This project focuses on optimising a key manufacturing process for a large industrial company, aiming to improve efficiency and profitability by reducing waste and maximising output. Using logistic regression to predict machine failure and analysing the local minima of the second derivative of machine performance, I derived operational thresholds to prevent failures and enhance performance.

The dataset contains information from 10,000 production sessions, including machine states and failure reasons. By identifying optimal operating thresholds, the analysis offers a clear understanding of when and why the machinery is likely to fail, allowing the company to take proactive steps to reduce the failure rate and increase productivity.

 What I Learned:

**Project Phases Matter**: I found that aiming to stick with the process—Task definition, Analysis, Development, and Testing—made the project run much smoother. Spending more time on defining the task and analysing what needs to be done upfront saved me from reworking during development and helped me move forward with clarity.

**Data Types Guide Model Selection**: Understanding the data types early on made model selection easier. Since I had independent variables that were numeric and binary, I knew that logistic regression would be a good fit for this classification task. Logistic regression is ideal for binary outcomes because it models the probability of an event occurring (such as machine failure) and outputs values between 0 and 1. This made it an obvious choice for predicting machine failure.

**Combining Statistics and Visuals**: I saw firsthand how powerful it can be to combine statistical modelling with clear visualisations. This combination allowed me to turn complex data into actionable insights that could directly support business decisions.

## Usage instructions

Run in the following order:
1. manufacturing_EDA.ipynb

If you want to test/play around with different data treatement techniques, plots and analysis:

2. preprocessing.py
3. analysis_and_visualisation.py


## File structure of the project

```
EDA_PROJECT/
│
├── config/
│   └── credentials.yaml          # Configuration file storing database credentials
│
├── data/
│   └── failure_data.csv          # The main dataset used for analysis
│
├── docs/
│   └── project_learnings.txt     # Documentation and learnings from the project
│
├── environments/
│   ├── environment.yml           # Conda environment file (basic setup)
│   └── environment_with_db.yml   # Conda environment file (with database utilities)
│
├── notebooks/
│   ├── db_utils.ipynb            # Jupyter notebook for database-related utilities
│   └── manufacturing_EDA.ipynb   # Main Jupyter notebook for exploratory data analysis
│   └── test.csv                  # Test file for quick data checks (if applicable)
│
├── requirements/
│   ├── requirements.txt          # Basic requirements file for pip installations
│   └── requirements_with_db.txt  # Requirements file with database-related packages
│
├── scripts/
│   ├── analysis_and_visualisation.py   # Script for analysis and visualization functions
│   ├── manufacturing_eda_classes.py    # Main class definitions for data processing
│   └── preprocessing.py                # Script for data preprocessing tasks
│
├── .gitignore                   # Git configuration to ignore unnecessary files
├── file_structure.png           # Image showing the project directory structure
└── README.md                    # Project overview and documentation

```

### Project Workflow
![alt text](<file_structure.png>)

## License information
Open to all 

