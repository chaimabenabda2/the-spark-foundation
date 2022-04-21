#!/usr/bin/env python
# coding: utf-8

# # Importing all libraries required in this notebook
# 

# In[46]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading data from remote link
# 

# In[47]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(10)


# In[5]:


data.shape


# # Plotting the distribution of scores
# 

# In[48]:


data.plot(x='Hours', y='Scores',style='o') 
plt.title('Hours vs score')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# make inputs and output

# In[10]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[12]:


X


# In[14]:


y


#  Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[15]:


from sklearn.model_selection import train_test_split  


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 


# # Training

# In[19]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


print("Training complete.")


# In[20]:


plt.scatter(X, y)


# In[24]:


regressor.coef_*X


# In[25]:


regressor.intercept_


# In[26]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# # **Making Predictions**
# Now that we have trained our algorithm, it's time to make some predictions.

# In[27]:


y_pred=regressor.predict(X_test)


# In[29]:


X_test


# In[30]:


y_pred


# In[31]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # Evaluating the mode

#  Mean Squared Error:is the Average of the square of the difference between actual and estimated values.

# In[45]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

