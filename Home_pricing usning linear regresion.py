#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import math


# In[3]:


home =pd.read_csv("E:\project\Home Pricing\home_data.csv")


# In[4]:


home.head()


# In[69]:


print("Total number of houses are: "+str(len(home)))


# In[7]:


home.columns


# In[8]:


plt.scatter(home.sqft_living,home.price)
plt.xlabel("sqft of living")
plt.ylabel("price")


# In[10]:


sns.lmplot("sqft_living","price",data = home)


# In[12]:


home.isnull().sum()


# In[15]:


sns.heatmap(home.isnull())


# In[16]:


print("No empty values")


# In[17]:


X = home[["bedrooms","bathrooms","sqft_living","sqft_lot","sqft_above","sqft_basement","floors","waterfront","view","condition","zipcode"]]
y = home[["price"]]


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)


# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


model = LinearRegression()


# In[24]:


model.fit(x_train, y_train)


# In[53]:


predict = model.predict(x_test)


# In[58]:


plt.scatter(y_test,predict)
plt.xlabel("actual price")
plt.ylabel("predicted price")


# In[59]:


model.coef_


# In[85]:


from sklearn import metrics


# In[135]:


mse = metrics.mean_squared_error(y_test, predict)


# In[137]:


rmse = np.sqrt(mse)
print("Root mean square for our first model is",rmse)


# In[144]:


model.fit(X,y)


# In[147]:


result= model.score(x_test, y_test)
print(("Accuracy: %.3f%%") % (result*100.0))

