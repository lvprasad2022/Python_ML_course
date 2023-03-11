#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import regex as re
from sklearn.model_selection import train_test_split


# In[5]:


df = pd.read_csv('C:/Users/PRASAD/OneDrive/Documents/GitHub/Python_ML_course/seattle-weather.csv')
df.head()


# In[6]:


df['weather'].value_counts()


# In[7]:


df.info()


# In[8]:


X = df.drop(columns=['date','weather'])
y = df['weather']
X.info()


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)
X_tr, X_cv, y_tr, y_cv = train_test_split(X_train,y_train,test_size=0.2,stratify=y_train)
X_tr.shape, X_cv.shape, y_tr.shape, y_cv.shape


# In[18]:


y_tr.value_counts()


# In[19]:


y_cv.value_counts()


# In[ ]:


import warnings
warnings.


# In[26]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

dtc = DecisionTreeClassifier(class_weight='balanced')
params = {
    'criterion' : ['gini','entropy'],
    'max_depth' : [None, 3,4,5,6,7],
    'min_samples_split' : [0.1,0.2,0.3,0.4,0.5],
    'min_samples_leaf' : [0.1,0.2,0.3,0.4,0.5]
}

clf_cv = RandomizedSearchCV(dtc,params,cv=10,scoring='f1_macro',error_score='raise')
clf_cv.fit(X_tr,y_tr)
print(clf_cv.best_params_)
print(clf_cv.best_score_)


# In[27]:


dtc = DecisionTreeClassifier(class_weight='balanced',min_samples_split=0.2,min_samples_leaf=0.1,max_depth=None,criterion='gini')
dtc.fit(X_tr,y_tr)
prediction = dtc.predict(X_cv)
y_df = pd.DataFrame()
y_df['Actual_Values'] = y_cv
y_df['Predicted_Values'] = prediction
print(y_df)


# In[28]:


from sklearn.metrics import classification_report

print(classification_report(y_df['Actual_Values'],y_df['Predicted_Values']))


# In[ ]:




