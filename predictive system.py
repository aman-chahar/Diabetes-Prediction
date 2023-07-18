# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('C:/Users/amanc/Desktop/Diabetes Prediction using Machine Learning/trained_model.sav', 'rb'))

input_data = (2,99,52,15,94,24.6,0.637,21)

#Changing the input data to a numpyarray
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("The person is not Diabetic")
else:
    print("The person is Diabetic")
    