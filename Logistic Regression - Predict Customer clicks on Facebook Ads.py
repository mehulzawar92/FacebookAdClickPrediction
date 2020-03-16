#!/usr/bin/env python
# coding: utf-8

# # Step 0 - Loading Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# # Step 1 - Loading Dataset

# In[2]:


training_set = pd.read_csv("Facebook_Ads_2.csv", encoding = 'ISO-8859-1')


# In[3]:


training_set.head(5)


# In[4]:


training_set.tail(5)


# # Step 2 - Explore/ Visualize Dataset

# In[5]:


clicked = training_set[training_set['Clicked'] == 1]

not_clicked = training_set[training_set['Clicked'] == 0]


# In[6]:


print('Total = ', len(training_set))
print('Number of Customers who clicked the Ad = ', len(clicked))
print('Number of Customers who did not click the Ad = ', len(not_clicked))
print('% of Customers who clicked the Ad = ', 1 * len(clicked)/ len(training_set) * 100, "%")
print('% of Customers who did not click the Ad = ', 1 * len(not_clicked)/ len(training_set) * 100, "%")


# In[7]:


sns.scatterplot(x = 'Time Spent on Site', y = 'Salary', data = training_set, hue = 'Clicked')


# In[8]:


sns.boxplot(x = 'Clicked', y = 'Salary', data = training_set)


# In[9]:


sns.boxplot(x = 'Clicked', y = 'Time Spent on Site', data = training_set)


# In[10]:


training_set.hist(bins = 40)


# In[11]:


training_set['Salary'].hist(bins = 40)


# # Step 3 - Preparing the data for Training / Cleaning Data

# In[12]:


training_set


# In[13]:


training_set.drop(['Names', 'emails'], axis = 1, inplace = True)


# In[14]:


training_set


# In[15]:


sns.heatmap(training_set.isnull(), yticklabels=False, cbar = False, cmap='Blues')


# In[16]:


#countries = pd.get_dummies(training_set['Country'], drop_first=True)


# In[17]:


#countries


# In[18]:


training_set.drop('Country', axis = 1, inplace = True)


# In[19]:


#training_set = pd.concat([training_set, countries], axis = 1)


# In[20]:


X = training_set.drop('Clicked', axis = 1).values


# In[21]:


y = training_set['Clicked'].values


# In[22]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# # Step 4 - Model Training

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


classifier = LogisticRegression(random_state = 0)


# In[27]:


classifier.fit(X_train, y_train)


# # Step 5 - Model Evaluation

# In[28]:


y_pred_train = classifier.predict(X_train)


# In[29]:


from sklearn.metrics import classification_report, confusion_matrix


# In[30]:


cm = confusion_matrix(y_train, y_pred_train)


# In[31]:


sns.heatmap(cm, annot = True, cbar = False, fmt = "d")


# In[32]:


print("Accuracy = ", (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]) * 100, "%")


# In[33]:


print(classification_report(y_train, y_pred_train))


# In[34]:


y_pred_test = classifier.predict(X_test)


# In[35]:


cm2 = confusion_matrix(y_test, y_pred_test)
print("Accuracy = ", (cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]) * 100, "%")


# In[36]:


sns.heatmap(cm2, annot = True, cbar = False, fmt = "d")


# In[37]:


print(classification_report(y_test, y_pred_test))


# # Step 6 - Visualising the data

# In[38]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

# Create a meshgrid ranging from the minimum to maximum value for both features

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))


# In[39]:


# plot the boundary using the trained classifier
# Run the classifier to predict the outcome on all pixels with resolution of 0.01
# Colouring the pixels with 0 or 1
# If classified as 0 it will be magenta, and if it is classified as 1 it will be shown in blue 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('lightblue', 'grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


# In[40]:


# plot all the actual training points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
    
plt.title('Facebook Ad: Customer Click Prediction (Training set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[41]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('lightblue', 'grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Training set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[42]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('lightblue', 'grey')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Testing set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:




