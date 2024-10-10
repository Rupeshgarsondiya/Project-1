'''
Author       : Rupesh Garsondiya
github       : @Rupeshgarsondiya
Organization : L.J University
'''

import streamlit as st
from train import *
from test import *


# Centering the title using HTML and CSS
st.markdown("<h1 style='text-align: center;'>Behavior classification on User behavior dataset</h1>", unsafe_allow_html=True)


class Main:
    def __init__(self) -> None:
        pass

    def run(self):
        t = test()
        t.predict_data()
        

m = Main()
m.run()