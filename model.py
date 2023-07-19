#!/usr/bin/env python
# coding: utf-8

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

# 读取excel文件， 指定 sheet_name
data_unique = pd.read_excel('./delear.xlsx', sheet_name='2', header=0)
# Hemodynamic deterioration 映射 wrosen：1， unworesen：0
target = data_unique['Hemodynamic deterioration'].map({'wrosen': 1, 'unwrosen': 0})
data_unique.insert(0, 'target', target)
# gender F:0, M:1
data_unique['gender'] = data_unique['gender'].map({'F':0, 'M':1})
# drop Hemodynamic deterioration and gender
data_unique.drop(['Hemodynamic deterioration'], axis=1, inplace=True)
data_unique['target'].value_counts()

# 取出可用特征及标签数据
feats = data_unique.iloc[:, 14:]
targets = data_unique.iloc[:, :2]
new_data_unique = pd.concat([targets, feats], axis=1)

imputer = KNNImputer(n_neighbors=3)
imputedpt = imputer.fit_transform(new_data_unique)
datafinal_imputed = pd.DataFrame(imputedpt, columns=new_data_unique.columns)

# feature_candi = pd.read_csv('./candi_feats.csv')

feature_candi = ['invasive_line','hypertension_disease', 'aki_stages','sbp_max','mbp_mean','bicarbonate_min','dbp_mean','temperature_mean','aniongap_max','urine_output']

features_select = datafinal_imputed[feature_candi]

sme = SMOTE(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(features_select, datafinal_imputed['target'], test_size=0.2, random_state=40, stratify=datafinal_imputed.target)

x_bsm, y_bsm = sme.fit_resample(X_train, y_train)
base_clf = XGBClassifier(colsample_bytree=0.3, gamma=0.01, learning_rate=0.1, max_depth=20, n_estimators=300)
train_set = (x_bsm, y_bsm)
eval_set = (X_test, y_test)

model = base_clf.fit(x_bsm, y_bsm)

print(model.score(X_test, y_test))

pickle.dump(model,open("clf.dat","wb"))

joblib.dump(model, 'model.pkl')