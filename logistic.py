import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DF = pd.read_csv("/Users/parmisazizi/Downloads/Coding Documents/ML and AI/titanic.csv")

print(DF.head())

#check missing values

print(DF.isnull().sum())

#data pre-processing

print('median of age column %.2f'% (DF["Age"].median(skipna = True)))
print('percent of missing records in cabin %.2f'%((DF["Cabin"].isnull().sum()/DF.shape[0]*100)))
print('Most common boarding point : %s' % DF['Embarked'].value_counts().idxmax())