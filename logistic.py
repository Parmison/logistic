import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

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

x = DF[["Pclass","Sex","Age","Fare","Embarked","travel alone"]]
y = DF[["Survived"]]

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2, random_state = 13)

LR_model = LogisticRegression()
LR_model.fit(xtrain,ytrain)

ypred = LR_model.predict(xtest)
matrix = confusion_matrix(ytest,ypred)
sns.heatmap(matrix,annot = True)

plt.title("confusion matrix")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

print(classification_report(ytest,ypred))