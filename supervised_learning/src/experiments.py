'''
@author: arehan7@gatech.edu
'''
#https://www.kaggle.com/uciml/iris
#https://www.kaggle.com/viraj19/habermancsv?select=haberman.csv

import pandas as pd
from sklearn.preprocessing.label import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import StandardScaler
from sklearn.ensemble.forest import RandomForestClassifier
df = pd.read_csv(r'C:\Users\AREHAN2\Documents\omscs\CS7641\iris.csv')
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
df.drop(labels='Id', inplace=True)
enc = LabelEncoder()
y = enc.fit_transform(df.Species)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#decision Tree
pipe = Pipeline([('scaler', StandardScaler()), 
                 #('label_enc', LabelEncoder()), 
                 ('dtc', DecisionTreeClassifier(min_samples_leaf=5))])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, y_train)
pipe.score(X_train, y_train)
pipe.score(X_test, y_test)

#random forest
rf_pipe = Pipeline([('scaler', StandardScaler()), 
                 #('label_enc', LabelEncoder()), 
                 ('rfc', RandomForestClassifier(n_estimators=20, n_jobs=4, random_state=0))])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
rf_pipe.fit(X_train, y_train)
rf_pipe.score(X_train, y_train)
rf_pipe.score(X_test, y_test)
