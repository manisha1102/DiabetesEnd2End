# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:19:35 2021

@author: manisha
"""

#importing the required libraries
import pandas as pd
import numpy as np
import pickle

#reading the diabetes dataset
df = pd.read_csv('D:/Datasets/diabetes.csv')
#Lets see how many rows and how many features are there
print('There are {} rows and {} columns'.format(df.shape[0], df.shape[1]))

df_copy = df.copy()
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.nan)

#df_copy['Pregnancies'].fillna(df_copy['Pregnancies'].median(), inplace=True)
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

#lets split the dataset into dependent and independent variables
x = df_copy.iloc[:, :-1]
y = df_copy.iloc[:, -1]

#lets split into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Now lets train our model
from sklearn.ensemble import RandomForestClassifier
lr = RandomForestClassifier(n_estimators=50)

lr.fit(x_train, y_train)

#pickle.dump(lr, open('model.pkl', 'wb'))

#model = pickle.load(open('model.pkl', 'rb'))

y_pred = lr.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))