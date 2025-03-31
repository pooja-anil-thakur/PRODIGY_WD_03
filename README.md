# PRODIGY_WD_03

# Bank Marketing Model(ML)

## Introduction
a machine learning model designed to predict customer responses to bank marketing campaigns. The model leverages a Decision Tree classifier to classify potential customers based on their demographic details and previous interactions with the bank.

## Dataset Description
The dataset used for this project is `bank-full.csv`, which includes various features related to customer demographics, employment details, financial information, and past campaign outcomes. The key columns in the dataset include:

- **age**: Age of the customer
- **job**: Type of job
- **marital**: Marital status
- **education**: Level of education
- **default**: Whether the customer has credit in default
- **balance**: Account balance
- **housing**: Whether the customer has a housing loan
- **loan**: Personal loan status
- **contact**: Contact communication type
- **day/month**: Last contact date
- **duration**: Last contact duration
- **campaign**: Number of contacts during this campaign
- **previous**: Number of previous contacts
- **poutcome**: Outcome of previous marketing campaigns
- **y**: Target variable (whether the client subscribed to a term deposit)

## Installation and Setup
To run this project, you need to have Python installed along with the necessary libraries. Install the dependencies using:

```bash
pip install pandas numpy scikit-learn
```

## Code Explanation

### 1. Load the Dataset

```python
import pandas as pd
df = pd.read_csv("bank-full.csv")
```

### 2. Data Preprocessing
Categorical variables are encoded using Label Encoding to make them suitable for machine learning algorithms:

```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['job'] = label_encoder.fit_transform(df['job'])
```

### 3. Splitting Data and Training the Model

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### 4. Evaluating the Model

```python
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Results
After training the model, the accuracy score provides an evaluation of its performance. Higher accuracy indicates better predictive capability. Example output:

```bash
Accuracy: 0.89
```

## Conclusion
This project demonstrates how a Decision Tree classifier can be used to predict customer responses to marketing campaigns. By analyzing demographic and previous interaction data, banks can target potential customers more effectively, improving campaign success rates. 




