#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier


# In[10]:


st.header("Hemodynamic Deterioration App for moderate-risk PE in ICU")
st.write("This is a web APP for identifying patients with moderate-risk pulmonary embolism who are predisposed to hemodynamic deterioration in the ICU.")

invasive_line=st.selectbox("invasive_line", ("1", "0"))
hypertension_disease=st.selectbox("hypertension", ("1", "0"))
aki_stages=st.selectbox("aki_stages", ("0", "1","2","3"))
sbp_max=st.sidebar.slider(label = 'sbp_max', min_value = 30.0,
                          max_value = 250.0 ,
                          value = 120.0,
                          step = 0.5)
mbp_mean=st.sidebar.slider(label = 'mbp_mean', min_value = 30.0,
                          max_value = 250.0 ,
                          value = 120.0,
                          step = 0.5)
bicarbonate_min=st.sidebar.slider(label = 'bicarbonate_min', min_value = 0.0,
                          max_value = 40.0 ,
                          value = 12.0,
                          step = 0.1)
dbp_mean=st.sidebar.slider(label = 'dbp_mean', min_value = 30.0,
                          max_value = 250.0 ,
                          value = 120.0,
                          step = 0.5)
temperature_mean=st.sidebar.slider(label = 'temperature_mean', min_value = 33.0,
                          max_value = 43.0 ,
                          value = 37.0,
                          step = 0.1)
aniongap_max=st.sidebar.slider(label = 'aniongap_max', min_value = 0.0,
                          max_value = 40.0 ,
                          value = 20.0,
                          step = 0.1)
urine_output=st.sidebar.slider(label = 'urine_output', min_value = 0.0,
                          max_value = 5000.0 ,
                          value = 2500.0,
                          step = 10.0)


# In[11]:


prob_xgb=np.array([9.50219035e-01, 7.61291608e-02, 3.33217345e-03, 2.31240392e-01,
       2.02291477e-02, 9.00495946e-02, 2.14633322e-03, 2.31014956e-02,
       1.26678275e-03, 5.74529208e-02, 2.71690078e-03, 3.04522306e-01,
       6.34758035e-03, 3.93795818e-02, 9.68539178e-01, 4.82272267e-01,
       4.22833830e-01, 9.12343897e-03, 1.63509712e-01, 7.45197246e-03,
       3.67531925e-03, 5.21537475e-02, 1.58926114e-01, 5.74302971e-01,
       8.35971534e-01, 2.76239842e-01, 1.44627085e-02, 1.15525581e-01,
       4.15479131e-02, 1.36471903e-02, 9.42359388e-01, 7.99663782e-01,
       6.98781967e-01, 9.03866366e-02, 1.23571316e-02, 1.72740340e-01,
       9.66963649e-01, 4.08647629e-03, 2.30748400e-01, 8.27927947e-01,
       4.77780163e-01, 3.76037449e-01, 2.00983416e-03, 9.39157978e-02,
       1.28159951e-02, 9.67270195e-01, 8.60421479e-01, 2.09921539e-01,
       4.18718308e-02, 2.08264659e-03, 4.32479717e-02, 6.89022690e-02,
       7.58193284e-02, 7.16732023e-03, 2.41674506e-03, 1.90761983e-02,
       1.18209265e-01, 2.74300086e-03, 7.83297122e-02, 1.06935762e-02,
       3.48192006e-02, 5.50516089e-03, 1.06827007e-03, 3.32793989e-03,
       8.76787364e-01, 1.86126737e-03, 2.97176205e-02, 5.23867039e-03,
       9.53180436e-03, 7.13162899e-01, 1.55708659e-02, 2.92127013e-01,
       2.90494841e-02, 1.98145583e-03, 1.82057999e-03, 5.47438800e-01,
       2.32683554e-01, 9.42107916e-01, 5.29685080e-01, 1.68804333e-01,
       2.79847346e-03, 6.48165063e-04, 2.99267843e-02, 8.31193873e-04,
       1.03962393e-02, 7.53100663e-02, 8.99955153e-01, 6.34902045e-02,
       6.16613701e-02, 5.19798040e-01, 2.36469343e-01, 2.51651108e-02,
       3.29087079e-02, 5.95742762e-02, 2.05094308e-01, 7.90839314e-01,
       2.55162711e-03, 1.81407213e-01, 2.04402115e-03, 2.57879263e-04,
       3.13793093e-01, 7.09664643e-01, 1.33047570e-02, 4.95273978e-01,
       1.52154118e-01, 1.17015734e-01, 1.12448046e-02, 7.13426899e-03,
       2.62574196e-01, 1.16437664e-02, 9.25292447e-02, 1.63948819e-01,
       2.62563135e-02, 5.70364064e-03, 1.80293564e-02, 1.39328182e-01,
       2.32648421e-02, 6.39662504e-01, 7.73392105e-03, 1.02363378e-02,
       2.02778960e-03, 1.15070934e-03, 1.33766085e-02, 2.05531403e-01,
       8.45564809e-03, 1.60080657e-04, 9.58191871e-04, 7.71062728e-03,
       5.79855312e-03, 7.37123549e-01, 6.68215007e-02, 3.31139177e-01,
       3.43803992e-03, 5.05046500e-03, 2.72601936e-02, 4.05492494e-03,
       9.38927904e-02, 1.95729770e-02, 3.88083644e-02, 4.55915974e-03,
       6.25861883e-01])


# In[12]:


y_test=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
       1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
       0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
       0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0])


# In[13]:


from sklearn.isotonic import IsotonicRegression


# In[14]:


Iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
Iso.fit(prob_xgb,y_test)


# In[15]:


if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load(open("model.pkl", "rb"))
    
    # Store inputs into dataframe
    X = pd.DataFrame([[invasive_line,hypertension_disease, aki_stages,sbp_max,mbp_mean,bicarbonate_min,dbp_mean,temperature_mean,aniongap_max,urine_output]], 
                     columns = ['invasive_line','hypertension_disease', 'aki_stages','sbp_max','mbp_mean','bicarbonate_min','dbp_mean','temperature_mean','aniongap_max','urine_output'])
    X[['invasive_line', 'hypertension_disease','aki_stages']] = X[['invasive_line', 'hypertension_disease','aki_stages']].astype('int64')
    # Get prediction
    X2=clf.predict_proba(X)[:, 1]
    print(X2)
    prediction =Iso.predict(X2)
    
    # Output prediction
    st.text(f"The probability of hemodynamic deterioration is as high as {prediction}")

