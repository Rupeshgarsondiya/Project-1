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
from sklearn.preprocessing import Binarizer,OrdinalEncoder,StandardScaler

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
        oe = OrdinalEncoder() # Ordinal Encoding to  convert categorical data into numerical data
        df[['P_Model','OS','Gender']] = oe.fit_transform(df[['P_Model','OS','Gender']])
        print(df.head())

        sc = StandardScaler() #  Standard Scaling to convert data into standard distribution
        df_update = sc.fit_transform(df.drop(columns = 'User Behavior Class'))

        # convert numpy array  to pandas dataframe
        df1 = pd.DataFrame(df_update,columns=['P_Model','OS','App_Time(hours/day)','Screen_time(hours/day)','Battery_Drain(mAh/day)','Installed_app'	,'Data_Usage(GB/day)','Age','Gender'])

        return df1,df['User Behavior Class']
