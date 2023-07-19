#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import streamlit as st
import pandas as pd
import joblib

st.header("Hemodynamic Deterioration App for moderate-risk PE in ICU")
st.write("This is a web APP for identifying patients with moderate-risk pulmonary embolism who are predisposed to hemodynamic deterioration in the ICU.")

invasive_line=st.selectbox("invasive_line", ("1", "0"))
hypertension_disease=st.selectbox("hypertension", ("1", "0"))
aki_stages=st.selectbox("aki_stages", ("0", "1","2","3"))
sbp_max=st.sidebar.slider(label = 'sbp_max', min_value = 30.0,
                          max_value = 250.0 ,
                          value = 120.0,
                          step = 5.0)
mbp_mean=st.sidebar.slider(label = 'mbp_mean', min_value = 30.0,
                          max_value = 250.0 ,
                          value = 120.0,
                          step = 5.0)
bicarbonate_min=st.sidebar.slider(label = 'bicarbonate_min', min_value = 0.0,
                          max_value = 40.0 ,
                          value = 12.0,
                          step = 1.0)
dbp_mean=st.sidebar.slider(label = 'dbp_mean', min_value = 30.0,
                          max_value = 250.0 ,
                          value = 120.0,
                          step = 5.0)
temperature_mean=st.sidebar.slider(label = 'temperature_mean', min_value = 33.0,
                          max_value = 43.0 ,
                          value = 37.0,
                          step = 0.2)
aniongap_max=st.sidebar.slider(label = 'aniongap_max', min_value = 0.0,
                          max_value = 40.0 ,
                          value = 20.0,
                          step = 1.0)
urine_output=st.sidebar.slider(label = 'urine_output', min_value = 0.0,
                          max_value = 5000.0 ,
                          value = 2500.0,
                          step = 50.0)

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load(open("model.pkl", "rb"))
    
    # Store inputs into dataframe
    X = pd.DataFrame([[invasive_line,hypertension_disease, aki_stages,sbp_max,mbp_mean,bicarbonate_min,dbp_mean,temperature_mean,aniongap_max,urine_output]], 
                     columns = ['invasive_line','hypertension_disease', 'aki_stages','sbp_max','mbp_mean','bicarbonate_min','dbp_mean','temperature_mean','aniongap_max','urine_output'])
    
    X[['invasive_line', 'hypertension_disease','aki_stages']] = X[['invasive_line', 'hypertension_disease','aki_stages']].astype('int64')
    print(X)
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"Based on feature values, this moderate_risk patients intend to be {prediction}")

