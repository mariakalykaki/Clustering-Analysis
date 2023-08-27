#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture


# In[2]:


data = pd.read_csv("Country-data.csv")
data.head()


# In[3]:


dataset = data.iloc[:,1:]
dataset.head()


# Finding optimal number of Clusters

# In[4]:


n_components = np.arange(1,10)
models = [GaussianMixture(n,random_state=1502).fit(dataset) for n in n_components]

plt.plot(n_components)


# In[5]:



model = GaussianMixture(4,random_state=1502).fit(dataset)


# Cluster Prediction by country

# In[6]:


cluster = pd.Series(model.predict(dataset))
cluster


# In[18]:


data["cluster"]=cluster
data.head()


# In[16]:


data.loc[data["country"]=="Greece"]


# In[25]:


data.loc[data["cluster"]== 2]


# In[ ]:




