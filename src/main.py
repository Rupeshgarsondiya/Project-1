'''
author         : Rupesh Garsondiya 
github       : @Rupeshgarsondiya
Organization : L.J University
'''


import pandas as pd
import numpy as np
from features.build_features import  *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


x = FeatureEngineering()  #  Creating an object of FeatureEngineering class
X ,y = x.get_clean_data() # get_clean_data method give clran data 


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print('2. Random Forest ')
print('----- Classification reaport -----')
print(classification_report(y_test,y_pred))
print('-----------------------')
print('-----------------------')
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred1 = rf.predict(x_test)
print('2. Random Forest ')
print('----- Classification reaport -----')
print(classification_report(y_test,y_pred1))





x1 = ['Xiaomi Mi 11','Android',154,4.0,761,32,322,42,'Male',5]
x.get_clean_data(x1)
print(lr.predict(x1))
    