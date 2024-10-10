''''
Author        : Rupesh Garsondiya
github        : @Rupeshgarsondiya
Organization  : L.J University

'''

import streamlit as st
import pandas as pd
import numpy as np
from train import *


class test:
    def __init__(self) -> None:
        pass
    def predict_data(self):
        mt = Model_Train()
        selected_algoritham = mt.train_model()

        options0 = ['Android','IOS']

        # Create the dropdown menu
        test_data = []
        with st.container():
         st.markdown('<div class="dropdown-left>', unsafe_allow_html=True)
         selected_option0 = st.selectbox('Select Oprating system  :', options0)
         st.markdown('</div>', unsafe_allow_html=True)

        options11 = ['Google Pixel 5', 'OnePlus 9','Samsung Galaxy S21','Xiaomi Mi 11']
        options22 = ['iPhone 12' ]

        # Create the dropdown menu
        
        with st.container():
         st.markdown('<div class="dropdown-left">', unsafe_allow_html=True)
         if selected_option0=='Android':
           selected_option11 = st.selectbox('Select Phone model name  :', options11)
         else:
           selected_option22 = st.selectbox('Select Phone model name  :', options22)
         st.markdown('</div>', unsafe_allow_html=True)

        options2 = ['MaP_Model	OS	App_Time(hours/day)	Screen_time(hours/day)	Battery_Drain(mAh/day)	Installed_app	Data_Usage(GB/day)	Age	Genderle','Female']

        # Create the dropdown menu
        
        with st.container():
         st.markdown('<div class="dropdown-left">', unsafe_allow_html=True)
         selected_option2 = st.selectbox('Select Gender  :', options2)

         st.markdown('</div>', unsafe_allow_html=True)



        
        no_of_app = st.number_input('Enter no of application  installed : ',max_value=100,min_value=10)
        
        hours_use = st.number_input('Enter no hours to use mobile  : ',max_value=23,min_value=0)

        data_use = st.number_input('Enter data use (GB) : ',min_value=0 ,max_value=10)

        app_time= st.number_input('Enter App use time (hours) : ',max_value=24,min_value=0)

        screen_time = st.number_input('Enter your screen time (hours) : ',max_value=24,min_value=0)

        with st.form(key =  " submit"):
         submit_button = st.form_submit_button(label='Submit')


        
        

        