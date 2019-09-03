#!/usr/bin/env python
# coding: utf-8

# # Ranking the Sensor based on its Predictive Power

# Ranking the Sensor based on its importance/predictive power
# with respect to the class labels of the samples is similar to rank the features of a dataset based on its important to prodict the correct output.
# 
# 

# # Data Preparation

# In[29]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
get_ipython().magic(u'pylab inline')

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV, SelectKBest

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

import warnings; warnings.simplefilter('ignore')


# In[30]:


df = pd.read_csv('task_data.csv')


# In[31]:


df.head()


# In[32]:


df = df.drop(['sample index'], axis = 1)
df.head()


# In[33]:


df.describe()


# In[34]:


df.isnull().sum()


# In[38]:


df.duplicated().sum()


# We do not have any null value and duplicate row in our Dataset.

# In[8]:


x, y = df.drop('class_label', axis=1), df['class_label']


# # Data Visualization

# In[20]:


sns.pairplot(data=df, hue='class_label')


# After analysing pairplot graph, We can see Sensor_6 is the most prdictive sensor among all others.
# 
# Other sensors like Sensor0, Sensor8 etc. are unable to predict class alone but are able to predict outcome if combine two sensors reading. As we can see from pair plot graph above sensor8 is able to predict outcome with sensor9 with less overlap of different class and so on. 

# In[14]:


# We visualize the first two principal components.
data = PCA(n_components=2).fit_transform(x)
temp = pd.DataFrame(data, columns=['a', 'b'])
temp['target'] = y
sns.lmplot('a', 'b', data=temp, hue='target', fit_reg=False)


# We can see from above graph that our data is balanced with less overlapping. Now we can check our model performance with different classification algorithms. 

# # Model training and Scoring

# In[21]:


classifiers = [('rfg', RandomForestClassifier(n_jobs=-1, criterion='gini')),
               ('rfe', RandomForestClassifier(n_jobs=-1, criterion='entropy')),
               ('extf', ExtraTreesClassifier(n_jobs=-1)),
               ('knn', KNeighborsClassifier(n_jobs=-1)),
               ('dt', DecisionTreeClassifier()),
               ('Et', ExtraTreeClassifier()),
               ('Logit', LogisticRegression()),
               ('gnb', GaussianNB()),
               ('bnb', BernoulliNB()),
              ]
allscores = []
for name, classifier in classifiers:
    scores = []
    for i in range(3): # three runs
        roc = cross_val_score(classifier, x, y, scoring='roc_auc', cv=20)
        scores.extend(list(roc))
    scores = np.array(scores)
    print(name, scores.mean())
    new_data = [(name, score) for score in scores]
    allscores.extend(new_data)
        


# In[22]:


temp = pd.DataFrame(allscores, columns=['classifier', 'score'])
sns.violinplot('classifier', 'score', data=temp, inner=None, linewidth=0.3)


# Every classifier has good performance except Bernoullie Navie Bias. Random Forest Classifier will be the best classifier here because it has high ROCAUC and low variation in the CV.

# In[23]:


classifier = RandomForestClassifier(n_jobs=-1, n_estimators=100)
rfecv = RFECV(estimator=classifier, cv=15, scoring='roc_auc')
rfecv.fit(x, y)
print("Optimal number of features : {}".format(rfecv.n_features_))


# As we get only one Optimal number of feature here using Feature ranking and cross-validated selection method which is Sensor6. We have already analyze it from our Pairplot graph above and all other sensor need atlest one another sensor reading to predict the class.

# In[27]:


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, '.-')
plt.plot([rfecv.n_features_, rfecv.n_features_], [0.9, 1], '-.')
plt.show()


# We can get the maximum score from one sensor reading only. We'll ranking all sensor with its predictive power using Ranking method of RFECV for each feature/Sensor.

# In[28]:


ranks = list(zip(rfecv.ranking_, x.columns))
ranks.sort()
ranks


# Woohoo!!!! We have our Sensor ranking ready based on its importance/predictive powerwith. Sensor6 is the most prdictive one and sensor7 is the least predictive powerwith. 

# In[ ]:




