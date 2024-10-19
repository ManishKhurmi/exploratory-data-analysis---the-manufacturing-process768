# exploratory-data-analysis---the-manufacturing-process768

## Installation instructions
run in the following order:
1. Run db_utils.ipynb


## Table of Contents, if the README file is long

## A description of the project: what it does, the aim of the project, and what you learned

This project focuses on optimising a key manufacturing process for a large industrial company, aiming to improve efficiency and profitability by reducing waste and maximising output. Using logistic regression to predict machine failure and analysing the local minima of the second derivative of machine performance, I derived operational thresholds to prevent failures and enhance performance.

The dataset contains information from 10,000 production sessions, including machine states and failure reasons. By identifying optimal operating thresholds, the analysis offers a clear understanding of when and why the machinery is likely to fail, allowing the company to take proactive steps to reduce the failure rate and increase productivity.

 What I Learned:

**Project Phases Matter**: I found that aiming to stick with the process—Task definition, Analysis, Development, and Testing—made the project run much smoother. Spending more time on defining the task and analysing what needs to be done upfront saved me from reworking during development and helped me move forward with clarity.

**Data Types Guide Model Selection**: Understanding the data types early on made model selection easier. Since I had independent variables that were numeric and binary, I knew that logistic regression would be a good fit for this classification task. Logistic regression is ideal for binary outcomes because it models the probability of an event occurring (such as machine failure) and outputs values between 0 and 1. This made it an obvious choice for predicting machine failure.

**Combining Statistics and Visuals**: I saw firsthand how powerful it can be to combine statistical modelling with clear visualisations. This combination allowed me to turn complex data into actionable insights that could directly support business decisions.

## Usage instructions

`brew install gfortran`

## File structure of the project
![alt text](<Screenshot 2024-10-19 at 09.04.24.png>)

## License information
Open to all 

ipython==8.12.3
matplotlib==3.9.2
numpy==1.26.0  # Use a stable version
pandas==2.2.3
scikit-learn==1.5.2  # Updated stable version
scipy==1.14.0
seaborn==0.12.2
statsmodels==0.14.0


attr==0.3.2
colorama==0.4.6
ConfigParser==7.1.0
cryptography==43.0.3
docutils==0.21.2
filelock==3.16.1
HTMLParser==0.0.2
ipython==8.12.3
ipywidgets==8.1.5
jnius==1.1.0
keyring==25.4.1
matplotlib==3.9.2
numpy==2.1.2
pandas==2.2.3
Pillow==11.0.0
protobuf==5.28.2
pyOpenSSL==24.2.1
redis==5.1.1
scikit_learn==1.4.2
scipy==1.14.1
seaborn==0.13.2
Sphinx==8.1.3
statsmodels==0.14.2
thread==2.0.5
urllib3_secure_extra==0.1.0
xmlrpclib==1.0.1

# From new env
A==1.0
argcomplete==3.5.1
astroid==3.3.5
asyncssh_unofficial==0.9.2
atheris==2.3.0
B==1.0.0
black==24.10.0
brotli==1.1.0
brotlicffi==1.1.0.0
cchardet==2.1.7
clr==1.0.3
colorama==0.4.6
ConfigParser==7.1.0
curio==1.6
docrepr==0.2.0
fqdn==1.5.1
gevent==24.10.3
gobject==0.1.0
h2==4.1.0
HTMLParser==0.0.2
hypothesis==6.115.3
importlib_metadata==7.1.0
importlib_resources==6.4.5
ipykernel==6.29.5
ipyparallel==8.8.0
ipywidgets==8.1.5
isoduration==20.11.0
jnius==1.1.0
js==1.0
jsonpointer==3.0.0
keyring==25.4.1
matplotlib==3.9.2
mock==5.1.0
netifaces==0.11.0
numpy==2.1.2
numpydoc==1.8.0
pandas==2.2.3
pathlib2==2.3.7.post1
Pillow==11.0.0
pkgutil_resolve_name==1.3.10
playwright==1.47.0
psutil==5.9.0
pycares==4.4.0
pycurl==7.45.3
pyczmq==0.0.4
pyodide==0.0.2
PyOpenGL==3.1.7
pyparsing==3.2.0
pyperf==2.8.0
PyQt4==4.11.4
PyQt5==5.15.11
PyQt6==6.7.1
PySide==1.2.4
PySide2==5.15.2.1
PySide6==6.8.0.1
pysqlite==2.8.3
rfc3339_validator==0.1.4
rfc3986_validator==0.1.1
rfc3987==1.3.8
scikit_learn==1.4.2
scipy==1.14.1
seaborn==0.13.2
setuptools==70.1.1
sip==6.8.6
Sphinx==8.1.3
statsmodels==0.14.2
testpath==0.6.0
trio==0.27.0
uri_template==1.3.0
urllib3_secure_extra==0.1.0
webcolors==24.8.0
win32security==2.1.0
xmlrpclib==1.0.1
yapf==0.40.2
zstandard==0.23.0
