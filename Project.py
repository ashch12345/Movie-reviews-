#!/usr/bin/env python
# coding: utf-8

# In[27]:


import nltk
from nltk import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import random


# In[28]:


cats = movie_reviews.categories()
reviews = []
for a in cats:
    for fid in movie_reviews.fileids(a):
        review = (list(movie_reviews.words(fid)),a)
        reviews.append(review)
random.shuffle(reviews)


# In[3]:


all_wd_in_reviews = nltk.FreqDist(wd.lower() for wd in movie_reviews.words())
top_wd_in_reviews = [list(wds) for wds in zip(*all_wd_in_reviews.most_common(2000))][0]


# In[4]:


def ext_ft(review,top_words):
    review_wds = set(review)
    ft = {}
    for wd in top_words:
        ft['word_present({})'.format(wd)] = (wd in review_wds)
    return ft


# In[5]:


featuresets = [(ext_ft(d,top_wd_in_reviews), c) for (d,c) in reviews]
train_set, test_set = featuresets[200:], featuresets[:200]


# In[33]:


from sklearn.metrics import accuracy_score
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


# In[7]:


classifier.show_most_informative_features(30)


# In[30]:


from sklearn.feature_extraction import DictVectorizer
dict_vectorizer=None
def get_train_test(train_set,test_set):
    global dict_vectorizer
    dict_vectorizer = DictVectorizer(sparse=False)
    X_train, y_train = zip(*train_set)
    X_train = dict_vectorizer.fit_transform(X_train)
    X_test,y_test = zip(*test_set)
    X_test = dict_vectorizer.transform(X_test)
    return X_train,X_test,y_train,y_test


# In[36]:


from sklearn.ensemble import RandomForestClassifier
X_train,X_test,y_train,y_test = get_train_test(train_set,test_set)
rf = RandomForestClassifier(n_estimators=100,n_jobs=4,random_state=10)
rf.fit(X_train,y_train)


# In[34]:


preds = rf.predict(X_test)
print(accuracy_score(y_test,preds))


# In[26]:


from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')
all_words_in_reviews = nltk.FreqDist(word.lower() for word in movie_reviews.words() if word not in stopwords_list)
top_words_in_reviews = [list(words) for words in zip(*all_words_in_reviews.most_common(2000))][0]


# In[31]:


featuresets = [(ext_ft(d,top_words_in_reviews), c) for (d,c) in reviews]
train_set, test_set = featuresets[200:], featuresets[:200]
X_train,X_test,y_train,y_test = get_train_test(train_set,test_set)


# In[32]:


rf = RandomForestClassifier(n_estimators=100,n_jobs=4,random_state=10)
rf.fit(X_train,y_train)


# In[23]:


preds = rf.predict(X_test)
print(accuracy_score(y_test,preds))


# In[22]:


features_list = zip(dict_vectorizer.get_feature_names(),rf.feature_importances_)
features_list = sorted(features_list, key=lambda x: x[1], reverse=True)
print(features_list[0:30])


# In[ ]:





# In[ ]:




