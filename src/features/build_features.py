'''
Author       : Rupesh Garsondiya
github       : @Rupeshgarsondiya
Organization : L.J university

'''

# Feature  Engineering

# import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import Binarizer,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline,make_pipeline
from  sklearn.compose import  ColumnTransformer


'''create class FeatureEngineering is  created to perform feature engineering on the dataset'''
class  FeatureEngineering:

    def __init__(self): # define  constructor
        pass

    def cleandata(self):
        data  = pd.read_csv('/home/rupeshgarsondiya/workstation/lab/Project-1/Data/user_behavior_dataset.csv') # load Dataset
       
        data.drop('User ID',axis=1,inplace=True)  # Drop user id column it not required

        '''Rename column name'''
        data.rename(columns={'Device Model':'P_Model','Operating System':'OS','App Usage Time (min/day)':'App_Time(hours/day)',
                   'Screen On Time (hours/day)':'Screen_time(hours/day)','Battery Drain (mAh/day)':'Battery_Drain(mAh/day)',
                   'Number of Apps Installed':'Installed_app','Data Usage (MB/day)':'Data_Usage(GB/day)'},inplace=True)
        
        # App time convert minit into the hours 
        data['App_Time(hours/day)']=data['App_Time(hours/day)']/60

        # convert data use MB into GB
        data['Data_Usage(GB/day)']=data['Data_Usage(GB/day)']/1024

        return data
    

    def get_clean_data(self):
        df  =  FeatureEngineering().cleandata()

        # train test split
        x_train,y_train,x_test,y_test = train_test_split(x = df.drop(columns=['User Behavior Class'],y = df['User Behavior Class'],test_size = 0.2))
        ct1 = ColumnTransformer([('P_Model',OneHotEncoder(),0),('oprating system',OneHotEncoder(),1),('age encode',OneHotEncoder(),8)],remainder='passthrough')
        ct2 = ColumnTransformer([('scaling battry(mah)',StandardScaler(),4)])

        #create sklearn pipline
        pipe = make_pipeline(ct1,ct2)
        pipe.fit(x_train)
        x_train = pipe.transform(x_train)
        x_test = pipe.transform(x_test)
        

        return x_train,y_train,x_test,y_test
    

        oe = OneHotEncoder(sparse_output= False) # Ordinal Encoding to  convert categorical data into numerical data
        categorical_columns = ['P_Model', 'OS', 'Gender']

        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)

        # Fit and transform the categorical data   
        encoded_data = encoder.fit_transform(df[categorical_columns])

        # Convert encoded data to a DataFrame and concatenate with the original dataset
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

        # Drop original categorical columns and add the encoded ones
        df_encoded = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

        # Show the first few rows of the updated dataset
        std = StandardScaler()
        df_std = std.fit_transform(df_encoded[['Battery_Drain(mAh/day)']])
        df1 = df_encoded.drop(columns=['User Behavior Class','Battery_Drain(mAh/day)'])
        df_final = pd.concat([df1,pd.DataFrame(df_std).rename(columns={0:'Battery_Drain(mAh/day)'})],axis=1)

        return df_final,df_encoded['User Behavior Class']
