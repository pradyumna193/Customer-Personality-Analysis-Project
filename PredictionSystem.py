# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:56:54 2023

@author: pavithra
"""

import numpy as np
import pickle
import pandas as pd

#load the model
classifier = pickle.load(open('D:/Deploying Machine Learning Model/classifier.pkl','rb'))


def segment_customers(input_data):

 prediction=classifier.predict(pd.DataFrame(input_data, columns=['Income','Kidhome','Teenhome','Age','Partner','Education_Level']))
 print(prediction)
 pred_1 = 0
 if prediction == 0:
     pred_1 = 'cluster 0'

 elif prediction == 1:
     pred_1 = 'cluster 1'

 elif prediction == 2:
    pred_1 = 'cluster 2'

 elif prediction == 3:
     pred_1 = 'cluster 3'

 return pred_1
