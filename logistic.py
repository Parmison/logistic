import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

DF = pd.read_csv("/Users/parmisazizi/Downloads/Coding Documents/ML and AI/titanic.csv")

print(DF.head())

#check missing values

print(DF.isnull().sum())

#data pre-processing

print('median of age column %.2f'% (DF["Age"].median(skipna = True)))
print('percent of missing records in cabin %.2f'%((DF["Cabin"].isnull().sum()/DF.shape[0]*100)))
print('Most common boarding point : %s' % DF['Embarked'].value_counts().idxmax())

#replacing nul values

DF["Age"].fillna(DF["Age"].median(skipna = True),inplace = True)
DF["Embarked"].fillna(DF['Embarked'].value_counts().idxmax(),inplace = True)
DF.drop("Cabin",axis = 1,inplace = True)

print(DF.isnull().sum())

#dropping unnecessary columns

DF.drop("Ticket",axis = 1,inplace = True)
DF.drop("Name",axis = 1,inplace = True)
DF.drop("PassengerId",axis = 1,inplace = True)
DF["travel alone"] = np.where((DF["SibSp"] + DF["Parch"]) > 0,0,1)
DF.drop("SibSp",axis = 1,inplace = True)
DF.drop("Parch",axis = 1,inplace = True)

print(DF.head())

#label encoding

label_encoder = preprocessing.LabelEncoder()

DF["Sex"] = label_encoder.fit_transform(DF["Sex"])
DF["Embarked"] = label_encoder.fit_transform(DF["Embarked"])

print(DF.head())