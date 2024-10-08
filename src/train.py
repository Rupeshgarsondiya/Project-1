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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report


class Model_Train:
    def __init__(self) -> None:
        pass

    '''load_data()  fuction use for to get the clean data or feature transformed data '''
    def load_data(self):
        data = FeatureEngineering() #  calling the class
        X,y = data.get_clean_data() #  load the clean data
        print(X.shape)
        print(y.shape)
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

        return  x_train,x_test,y_train,y_test  # perform train test split
    
    '''This function use to train model '''
    def train_model(self):
        x_train,x_test,y_train,y_test = self.load_data() # calling load_data function to load data
        '''Here so many option for user  to select the model user selected model train  and genrate the classification
        report '''
        while True:
            print("Enter 1 for Logistic Regreesion : ")
            print("Enter 2 for Random Forest Classifier : ")
            print("Enter 3 for Decision Tree Classifier : ")
            print("Enter 4 for  K-Nearest Neighbors : ")
            print("Enter 5 for  Support Vector Machine : ")
            print("Enter 6 for  Exit : ")
            print()
            a = int(input('Enter number which model you want to be train : '))
            if a ==1:
                lr = LogisticRegression()
                lr.fit(x_train,y_train)
                y_pred = lr.predict(x_test)
                print('----------Logistic Regreesion-----------')
                print(classification_report(y_test,y_pred))
            elif a ==2:
                rf = RandomForestClassifier(n_estimators=20,max_depth=3,n_jobs=-1,max_features=5)
                rf.fit(x_train,y_train)
                y_pred1 = rf.predict(x_test)
                print('---------Random Forest Classifier---------')
                print(classification_report(y_test,y_pred1))
            elif a==3:
                dt = DecisionTreeClassifier(max_depth=3)
                dt.fit(x_train,y_train)
                y_pred2 = dt.predict(x_test)
                print('---------Decision Tree Classifier---------')
                print(classification_report(y_test,y_pred2))
            elif a ==4:
                knn = KNeighborsClassifier()
                knn.fit(x_train,y_train)
                y_pred3 = knn.predict(x_test)
                print('---------K-Nearest Neighbors---------')
                print(classification_report(y_test,y_pred3))
            elif a==5:
                svm = SVC()
                svm.fit(x_train,y_train)
                y_pred4 = svm.predict(x_test)
                print('---------Support Vector Machine---------')
                print(classification_report(y_test,y_pred4))
            elif a==6:
                print()
                print()
                print('----THANK YOU-----')
                break




tm = Model_Train()
tm.train_model()







