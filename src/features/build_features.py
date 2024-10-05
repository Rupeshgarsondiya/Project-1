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
from sklearn.preprocessing import Binarizer,LabelEncoder
from notebook import eda



#print(df.head()) # display first 5 rows of dataset

#print(df.info()) # display information about dataset


le = LabelEncoder()
eda.df['Gender'] = le.fit_transform(eda.df['Gender']) # To encode cetegorical column  1 = Male and 2 = Female
eda.df['OS'] = le.fit_transform(eda.df['OS'])
