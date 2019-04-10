#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk 


# In[2]:


from nltk.corpus import names


# In[3]:


word='sherlock'


# In[5]:


def gender_features(word):
    return{'last_letter':word[-1]}


# In[6]:


gender_features(word)


# In[9]:


labeled_names=([(name,'male') for name in names.words('male.txt')]+[(name,'female') for name in names.words('female.txt')])


# In[10]:


len(names.words())


# In[11]:


labeled_names


# In[12]:


import random


# In[13]:


random.shuffle(labeled_names)


# In[14]:


featuresets=[(gender_features(n),gender) for (n,gender) in labeled_names]


# In[15]:


featuresets


# In[18]:


train_set,test_set = featuresets[500:], featuresets[:500]


# In[20]:


classifier=nltk.NaiveBayesClassifier.train(train_set)


# In[23]:


classifier.classify(gender_features('David'))


# In[24]:


classifier.classify(gender_features('Michelle'))


# In[25]:


print(nltk.classify.accuracy(classifier,test_set))


# In[ ]:




