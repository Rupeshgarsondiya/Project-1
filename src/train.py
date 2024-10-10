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



class Model_Train:
    def __init__(self) -> None:
        pass

    '''load_data()  fuction use for to get the clean data or feature transformed data '''
    def load_data(self):
        data = FeatureEngineering() #  calling the class
        return  data.get_clean_data()
    
    '''This function use to train model '''
    def train_model(self):
        x_train,x_test,y_train,y_test = self.load_data() # calling load_data function to load data

        # Define the options for the dropdown menu
        options = ['Logistic Regreesion', 'Random Forest Classifier', 'Decision Tree', 'SVM','KNeighborsClassifier']

        # Create the dropdown menu
        
        with st.container():
         st.markdown('<div class="dropdown-left">', unsafe_allow_html=True)
        selected_option = st.selectbox('Select Algoritham :', options)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display the selected option
        st.write(f'You selected: {selected_option}')

        if selected_option== 'Logistic Regreesion':
            lr = LogisticRegression()
            lr.fit(x_train,y_train)
            ypred = lr.predict(x_test)
        elif selected_option=='Random Forest Classifier':
            rf = RandomForestClassifier()
            rf.fit(x_train,y_train)
            ypred1 = rf.predict(x_test)
        elif selected_option=='Decision Tree':
            dt = DecisionTreeClassifier()
            dt.fit(x_train,y_train)
            ypred2 = dt.predict(x_test)
        elif selected_option =='SVM':
            svm = SVC()
            svm.fit(x_train,y_train)
            ypred3 = svm.predict(x_test)
        elif selected_option=='KNeighborsClassifier':
            knn = KNeighborsClassifier()
            knn.fit(x_train,y_train)
            ypred4  = knn.predict(x_test)           
        else:
            pass

        return selected_option
