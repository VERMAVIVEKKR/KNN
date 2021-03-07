# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 08:55:43 2021

@author: Vivek
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score


glass =pd.read_csv("E:\\8sept\\assigement\\KNN//glass.csv")

glass.head()

# value count for glass types
glass.Type.value_counts()

#Data exploration and visualizaion

#correlation matrix -
cor = glass.corr()
sns.heatmap(cor)

#Scatter plot of two features, and pairwise plot

sns.scatterplot(glass['RI'],glass['Na'],hue=glass['Type'])

#pairwise plot of all the features
sns.pairplot(glass,hue='Type')
plt.show()

#Feature Scaling

scaler = StandardScaler()
scaler.fit(glass.drop('Type',axis=1))


StandardScaler(copy=True, with_mean=True, with_std=True)


#perform transformation
scaled_features = scaler.transform(glass.drop('Type',axis=1))
scaled_features

glass = glass(scaled_features,columns=glass.columns[:-1])
glass.head()

dff = glass.drop(['Ca','K'],axis=1) #Removing features - Ca and K 
X_train,X_test,y_train,y_test  = train_test_split(dff,glass['Type'],test_size=0.3,random_state=45) #setting random state ensures split is same eveytime, so that the results are comparable

knn = KNeighborsClassifier(n_neighbors=4,metric='manhattan')
knn.fit(X_train,y_train)

knn.fit(X_train,y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=4, p=2,
                     weights='uniform')

    y_pred = knn.predict(X_test)
    print(classification_report(y_test,y_pred))
    accuracy_score(y_test,y_pred)
