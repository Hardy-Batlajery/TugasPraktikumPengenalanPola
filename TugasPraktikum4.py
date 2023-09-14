#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')


# In[75]:


data = pd.read_csv("tugas/train.csv")
data.head()


# In[76]:


X = data [["text"]]
y = data [["label"]]

print(X)
print(y.head())


# In[77]:


X = X.astype(str)


# In[78]:


X = X["text"].str.lower()


# In[79]:


print(X.head())


# In[80]:


PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

X = X.apply(lambda text: remove_punctuation(text))
X.head()


# In[82]:


STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

X = X.apply(lambda text: remove_stopwords(text))
X.head()


# In[83]:


cnt = Counter()
for text in X.values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)


# In[13]:


FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

X = X.apply(lambda text: remove_freqwords(text))
X.head()


# In[84]:


n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

X = X.apply(lambda text: remove_rarewords(text))
X.head()


# In[85]:


lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

X = X.apply(lambda text: lemmatize_words(text))
print (X.head())


# In[86]:


X = X.values
vectorizer = CountVectorizer(binary=True)
features = vectorizer.fit_transform(X)


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=3)


# In[88]:


model = BernoulliNB()
model.fit(X_train, y_train)


# In[24]:


train = model.predict(X_train)
cm = confusion_matrix(y_train, train)
print(classification_report(y_train ,train))
print("Confusion Matrix: \n", cm)
print("F1 Score: ",f1_score(y_train, train))
print("Accuracy: ", accuracy_score(y_train, train))
print("Precision: ", precision_score(y_train, train))
print("Recall: ", recall_score(y_train, train))


# In[89]:


test = model.predict(X_test)
cm = confusion_matrix(y_test, test)
print(classification_report(y_test ,test))
print("Confusion Matrix: \n", cm)
print("F1 Score: ",f1_score(y_test, test))
print("Accuracy: ", accuracy_score(y_test, test))
print("Precision: ", precision_score(y_test, test))
print("Recall: ", recall_score(y_test, test))


# In[ ]:




