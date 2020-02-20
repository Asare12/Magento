#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pydotplus


# In[2]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


df = pd.read_csv("used_car_data_2.csv")


# In[4]:


df_2 = df.drop(['title_status','model','cylinders','fuel','transmission','vin','drive','size','paint_color','image_url','description','county','state','lat','long'], axis = 'columns')


# In[5]:


df_2.head(5)


# In[6]:


output = df_2['price']
output


# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


manufacturer_label = LabelEncoder()
condition_label = LabelEncoder()
type_label = LabelEncoder()


# In[9]:


df_2['manufacturer_n'] = manufacturer_label.fit_transform(df_2['manufacturer'])
df_2['condition_n'] = condition_label.fit_transform(df_2['condition'])
df_2['type_n'] = type_label.fit_transform(df_2['type'])


# In[11]:


#df_2.head(5)
#df_2.to_csv('dummies_67%.final.csv', index=False)


# In[12]:


manufacture_df = df_2.drop(['condition','odometer','year','price','condition_n','type','type_n'], axis = 'columns')
manufacture_df.drop_duplicates()


# In[13]:


type_df = df_2.drop(['condition','odometer','year','price','condition_n','manufacturer','manufacturer_n'], axis = 'columns')
type_df.drop_duplicates()


# In[14]:


condition_df = df_2.drop(['odometer','year','price','manufacturer','manufacturer_n','type','type_n'], axis = 'columns')
condition_df.drop_duplicates()


# In[15]:


inputs = final_df = df_2.drop(['price','condition','type','manufacturer'], axis = 'columns')


# In[16]:


inputs.head(5)


# In[17]:


len(output) - len(inputs)


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(inputs,output, test_size = 0.2, random_state = 42)


# In[20]:


#feature_scaling
##from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.fit_transform(x_test)


# In[21]:


#min-max scaling
#from sklearn.preprocessing import MinMaxScaler
#msc = MinMaxScaler()
#x_train = msc.fit_transform(x_train)
#x_test = msc.fit_transform(x_test)


# # Decision Tree Regression

# In[22]:


from sklearn import tree


# In[23]:


tree_regression_model = tree.DecisionTreeRegressor()


# In[24]:


tree_regression_model.fit(x_train,y_train)


# In[25]:



tree_regression_model.score(x_test,y_test)*100


# In[26]:


year =2018
mileage =22000
manufacturer =1
condition =0
car_type =0

price = tree_regression_model.predict([[year,mileage,manufacturer,condition,car_type]])

price_ranger = price*0.2
price_min  = price-price_ranger
#price_max = price+price_ranger

print(price)


# # RandomForestRegression

# In[33]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(x_train, y_train)


# In[28]:


import numpy as np


# In[29]:


regressor.score(x_test,y_test)*100


# In[30]:


year =2018
mileage =22000
manufacturer =1
condition =0
car_type =0

price = regressor.predict([[year,mileage,manufacturer,condition,car_type]])
print("$",price)


# In[31]:


year =2060
mileage =22
manufacturer =6
condition =4
car_type =8


price = regressor.predict([[year,mileage,manufacturer,condition,car_type]])
print("$", price)


# In[ ]:




