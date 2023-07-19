#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
import joblib

import warnings
warnings.filterwarnings('ignore')


data_unique=pd.read_csv('C:/Users/Cola/Desktop/MIMIC1/selection_feature.csv')



sme = SMOTE(random_state=42)

X=data_unique.iloc[:,2:]#x的范围是第一列到最后一列
y=data_unique["target"]#括号内是最后一列的名称
X_train, X_test, y_train, y_test = train_test_split(data_unique, data_unique['target'], test_size=0.2, random_state=82, stratify= data_unique.target)
x_train = X_train.iloc[:, 2:]
x_test = X_test.iloc[:, 2:]
x_bsm, y_bsm = sme.fit_resample(x_train, y_train)



model = XGBClassifier(colsample_bytree=0.3, gamma=0.01, learning_rate=0.1, max_depth=20, n_estimators=300)
model.fit(x_bsm, y_bsm)


# In[2]:


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression


y_pred_proba = model.predict_proba(x_test)[:, 1]

# 计算AUC得分
model_auc = roc_auc_score(y_test,y_pred_proba)
print("AUC score: %.3f" % model_auc)


# In[5]:


joblib.dump(model, 'model.pkl')
