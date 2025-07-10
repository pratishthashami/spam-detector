#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


# import dataset

spam_df = pd.read_csv(r'C:\Users\user\Downloads\emails.csv')
spam_df


# In[5]:


spam_df.describe()


# In[6]:


spam_df.info()


# ## Visualize Dataset

# In[8]:


ham = spam_df[spam_df['spam']==0]
ham


# In[9]:


spam = spam_df[spam_df['spam']==1]
spam


# In[10]:


print('Spam percentage -', len(spam)/len(spam_df)*100,'%')


# In[12]:


print('Ham percentage - ', len(ham)/len(spam_df)*100,'%')


# In[21]:


sns.countplot(x='spam', data=spam_df)
plt.xlabel('Spam (1) vs Ham (0)')
plt.ylabel('Count')
plt.title('Count of Spam vs Ham')
plt.show()


# ## Count Vectorizer Example

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer


# In[30]:


sample_data=['This is the first document.','This document is the second document', 'And this is the third document.','Is this the fourth document']

sample_vectorizer = CountVectorizer()


# In[33]:


x= sample_vectorizer.fit_transform(sample_data)


# In[34]:


print(x.toarray())


# In[35]:


print(sample_vectorizer.get_feature_names())


# ## Apply count vectorizer to the spam/ham dataset

# In[36]:


from sklearn.feature_extraction.text import CountVectorizer


# In[38]:


vectorizer = CountVectorizer()
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])


# In[39]:


print(vectorizer.get_feature_names())


# In[41]:


print(spamham_countvectorizer.toarray())


# In[43]:


spamham_countvectorizer.shape


# ## Training the model

# In[45]:


label = spam_df['spam'].values


# In[46]:


label


# In[44]:


from sklearn.naive_bayes import MultinomialNB


# In[49]:


NB_classifier = MultinomialNB()
NB_classifier.fit(spamham_countvectorizer,label)
print(f'MultinomialNB(alpha={NB_classifier.alpha}, class_prior={NB_classifier.class_prior}, fit_prior={NB_classifier.fit_prior})')


# In[52]:


testing_sample1=['Free Money!','Hi Jim! Let me know if any info required.']
testing_sample_countvectorizer= vectorizer.transform(testing_sample1)


# In[53]:


NB_classifier.predict(testing_sample_countvectorizer)


# In[54]:


testing_sample2=['we can discuss it together','money viagara!!!']
testing_sample2_countvectorizer= vectorizer.transform(testing_sample2)


# In[55]:


NB_classifier.predict(testing_sample2_countvectorizer)


# ## Divide the data into training and testing

# In[56]:


X= spamham_countvectorizer
y=label


# In[58]:


X.shape


# In[59]:


y.shape


# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[61]:


from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# ## Evaluating the Model

# In[63]:


from sklearn.metrics import confusion_matrix, classification_report


# In[64]:


y_predict_train = NB_classifier.predict(X_train)
y_predict_train


# In[66]:


cm= confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[67]:


y_predict_test = NB_classifier.predict(X_test)
cm= confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[68]:


print(classification_report(y_test, y_predict_test))


# In[ ]:




