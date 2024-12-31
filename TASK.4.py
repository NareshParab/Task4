#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Data Generation ,a large dataset of 50,000 records
np.random.seed(42)

n_samples = 50000  # Large dataset of 50,000 records

# Simulated features: Age, Income, Browsing Time (minutes), Previous Purchase (Yes/No)
data = {
    'Age': np.random.randint(18, 70, n_samples),
    'Income': np.random.randint(25000, 150000, n_samples),
    'Browsing Time': np.random.randint(5, 120, n_samples),  # Browsing time in minutes
    'Previous Purchase': np.random.choice(['Yes', 'No'], n_samples),
    'Purchased': np.random.choice([0, 1], n_samples)  # Target variable (1 = Purchased, 0 = Not Purchased)
}

# Create DataFrame
df = pd.DataFrame(data)

# Data Preprocessing
# Encoding categorical variables (e.g., Previous Purchase)
le = LabelEncoder()
df['Previous Purchase'] = le.fit_transform(df['Previous Purchase'])

# Features and target variable
X = df.drop('Purchased', axis=1)  # Features (demographic and behavioral data)
y = df['Purchased']  # Target variable (whether the customer made a purchase)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test)

# Convert numerical predictions (0 or 1) --> "Yes" or "No"
y_pred = ["Yes" if prediction == 1 else "No" for prediction in y_pred]

# Accuracy and classification report
accuracy = accuracy_score(y_test, clf.predict(X_test))
report = classification_report(y_test, clf.predict(X_test), target_names=["No", "Yes"])

# Results
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Output all predictions for the test set
print("\nAll Predictions (Purchase or Not):")
for i in range(len(y_pred)):
    print(f"Customer {i+1}: {y_pred[i]}")


# In[ ]:




