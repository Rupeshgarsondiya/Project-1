'''
author         : Rupesh Garsondiya 
github       : @Rupeshgarsondiya
Organization : L.J University
'''





import pandas as pd
import streamlit as st
import numpy as np
from features.build_features import  *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score




st.markdown(
    """
    <style>
    /* Change the background color of the entire page */
    .stApp {
        background-color: #f0f2f6;  /* Set your desired color here */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title('Welcome  !')
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

        st.write("- Enter 1 for Logistic Regreesion : ")
        st.write("- Enter 2 for Random Forest Classifier : ")
        st.write("- Enter 3 for Decision Tree Classifier : ")
        st.write("- Enter 4 for  K-Nearest Neighbors : ")
        st.write("- Enter 5 for  Support Vector Machine : ")
        st.write("- Enter 6 for  Exit : ")
        print()            
        a = st.number_input('-> Enter number which model  you want to train : ',min_value=1,max_value=5,placeholder = 'Enter number',value=None)
        if a ==1:
            lr = LogisticRegression()
            lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            st.write('----------Logistic Regreesion-----------')
            st.write('Accuracy Score',accuracy_score(y_test,y_pred))
        elif a ==2:
            rf = RandomForestClassifier(n_estimators=20,max_depth=3,n_jobs=-1,max_features=5)
            rf.fit(x_train,y_train)
            y_pred1 = rf.predict(x_test)
            st.write('---------Random Forest Classifier---------')
            st.write('Accuracy Score',accuracy_score(y_test,y_pred1))
        elif a==3:
            dt = DecisionTreeClassifier(max_depth=3)
            dt.fit(x_train,y_train)
            y_pred2 = dt.predict(x_test)
            st.write('---------Decision Tree Classifier---------')
            st.write('Accuracy Score',accuracy_score(y_test,y_pred2))
        elif a ==4:
            knn = KNeighborsClassifier()
            knn.fit(x_train,y_train)
            y_pred3 = knn.predict(x_test)
            st.write('---------K-Nearest Neighbors---------')
            st.write('Accuracy Score',accuracy_score(y_test,y_pred3))            
        elif a==5:
            svm = SVC()
            svm.fit(x_train,y_train)
            y_pred4 = svm.predict(x_test)
            st.write('---------Support Vector Machine---------')
            st.write('Accuracy Score',accuracy_score(y_test,y_pred4))
        elif a==6:
            print()
            print()
            st.write('----THANK YOU-----')
        else:
            print('Invalid input !')



tm = Model_Train()
tm.train_model()






