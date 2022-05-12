
#%%
import pandas as pd
import numpy as np
import warnings
import random
import time 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from cleaner import start_cleaning, get_cyberbully_prob

warnings.filterwarnings("ignore")

start_time = time.time()

seed_value = 20230337
random.seed(seed_value)
np.random.seed(seed_value)

df = pd.read_csv("cyberbullying_tweets.csv")
df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})

# find duplicated ones and clear them
df.duplicated().sum()
df = df[~df.duplicated()]

# show categorical data counts
df.sentiment.value_counts()

texts_new = start_cleaning(df)

df['tweet_clean'] = texts_new
df.head()
df["tweet_clean"].duplicated().sum()
# clean duplicate data
df.drop_duplicates("tweet_clean", inplace=True)
# show sentiment counts
df.sentiment.value_counts()

# removing other_cyberbullying categories will improve performance (%74 to %83)
# because all the tweets in that category is actually contains the tweets that
# can be any of the other categories (age, ethnicity, religion, gender)
df = df[df["sentiment"]!="other_cyberbullying"]
sentiments = ["religion", "age", "ethnicity", "gender", "not bullying"]

# get word count for tweets
tweet_length = []
for text in df.tweet_clean:
    tweet_len = len(text.split())
    tweet_length.append(tweet_len)
df['tweet_length'] = tweet_length
# TODO: a word must contain at least 4 letters. this can be tested with 3
df = df[df['tweet_length'] > 3]

# number of letter
df.sort_values(by=['tweet_length'], ascending=False)
# limit chars to 120
# TODO: must try 80, 100
df = df[df['tweet_length'] < 120]
max_length = np.max(df['tweet_length'])

# convert sentiments to numbers 
df['sentiment'] = df['sentiment'].replace(
    {
        'religion':0,
        'age':1,
        'ethnicity':2,
        'gender':3,
        'not_cyberbullying':4
    }
)

# split the data into test and train 
X = df['tweet_clean']
y = df['sentiment']
# train and test splitting: 
# %20 -> Test
# %80 -> Train
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=seed_value
)

# Multinominal Naive Bayes
# create token matrix with tweets
count_vectorizer = CountVectorizer()
X_train_cv =  count_vectorizer.fit_transform(X_train)
X_test_cv = count_vectorizer.transform(X_test)

# TF-IDF: Term Frequency-Inverse Document Frequency
# shows the importance of a word in the document
#       number of times the word appears in the document 
# TF = ---------------------------------------------------
#           total number of words in the document
#
#                  number of documents in the corpus
# IDF = log(-----------------------------------------------) 
#            number of documents that contains the word +1
#
# TF-IDF = TF * IDF
# TODO: can be improved with other params.
term_freq_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
X_train_tf = term_freq_transformer.transform(X_train_cv)
X_test_tf = term_freq_transformer.transform(X_test_cv)

# TODO: can be tested with other params, (alpha value)
multinominal_naive_bayes = MultinomialNB()
multinominal_naive_bayes.fit(X_train_tf, y_train)
prediction_results = multinominal_naive_bayes.predict(X_test_tf)

accuracy_percentage = classification_report(
    y_test, 
    prediction_results, 
    target_names = sentiments,
    output_dict = True
)['accuracy']*100

print(
    'Algorithm : Naive Bayes\n',
    classification_report(
        y_test,
        prediction_results,
        target_names = sentiments,
        labels = [0, 1, 2, 3, 4]
    )
)
print(f"Elapsed time in seconds for Naive-Bayes:  {round(time.time()-start_time, 2)}s")
print(
    f"Considering the tweets of the user, \n"
    f"it was decided that this user is a cyberbully with:\n"
    f"Probability\t{get_cyberbully_prob(prediction_results)}%\n"
    f"Accuracy\t{round(accuracy_percentage, 2)}%\n"
)

#
#  ___                   _  __                 _    
# / _ \ ______ _ _ __   | |/ /___  _   _ _   _| | __
#| | | |_  / _` | '_ \  | ' // _ \| | | | | | | |/ /
#| |_| |/ / (_| | | | | | . \ (_) | |_| | |_| |   < 
# \___//___\__,_|_| |_| |_|\_\___/ \__, |\__,_|_|\_\
#                                  |___/            
#