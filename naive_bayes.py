
#%%
import pandas as pd
import numpy as np
import re, string
import emoji
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import random
import time 

seed_value = 20230337
random.seed(seed_value)
np.random.seed(seed_value)
# stop_words = [..., 'can', 'did', 'am', 'is', 'are', ...]
stop_words = set(stopwords.words('english'))

df = pd.read_csv("cyberbullying_tweets.csv")
# df.head()
# df.info()
df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})

# find duplicated ones and clear them
df.duplicated().sum()
df = df[~df.duplicated()]
# df.info()

# show categorical data counts
df.sentiment.value_counts()

# remove all punctuation and emojis
def clear_punctuation(text): 
    # remove punctuation chars
    punct_chars = string.punctuation
    table = str.maketrans('', '', punct_chars)
    text = text.translate(table)

    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    return emoji.replace_emoji(text, replace='')

# remove contractions
# TODO: improve contractions, can be used a library, improve regex searches, clearing data can be better
def clear_with_regex(text):
    text = text.replace('\r', '').replace('\n', ' ').lower() # remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) # remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) # remove non utf8/ascii
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub("\s\s+" , " ", text)
    return text

# remove # symbol only
def clean_hashtags(tweet):
    # remove hashtags from the end of the tweet.
    cleared_tweet_ending = " ".join(
        word.strip() 
        for word in re.split(
            '#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', 
            tweet
            )
        )
    
    # remove hashtags from the middle of the tweet
    all_clear = " ".join(
        word.strip() 
        for word in re.split(
            '#|_', 
            cleared_tweet_ending
            )
        ) 
    return all_clear

# remove special words like money, percentages etc.
def clear_special_chars(_tweet):
    sent = []
    special_chars = ['$', '€', '&', '%']
    for word in _tweet.split(' '):
        if any(special_char in word for special_char in special_chars):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

# tokenize and remove morphological affixes from words
def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    # TODO: maybe try SnowballStemmer()
    ps = PorterStemmer()
    return ' '.join([ps.stem(word) for word in tokenized])

# clear every tweet with this order
# TODO: can be improved later on
def clean_tweet(text):
    text = clear_punctuation(text)
    text = clear_with_regex(text)
    text = clean_hashtags(text)
    text = clear_special_chars(text)
    text = stemmer(text)
    return text

print('Calculation starting now..')
start_time = time.time()

texts_new = []
for _tweet in df.text:
    texts_new.append(clean_tweet(_tweet))

df['tweet_clean'] = texts_new
df.head()
df["tweet_clean"].duplicated().sum()
# clean duplicate data
df.drop_duplicates("tweet_clean", inplace=True)
df.shape
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

# train and validation splitting :
# convert %80 of train data into : 
# %10 -> Validation 
# %90 -> Train
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, 
    y_train, 
    test_size=0.1, 
    stratify=y_train, 
    random_state=seed_value
)

# Multinominal Naive Bayes
# create token matrix with tweets
count_vectorizer = CountVectorizer()
X_train_cv =  count_vectorizer.fit_transform(X_train)
X_test_cv = count_vectorizer.transform(X_test)

# convert term-freq with inverse-document-frequency
# TODO: can be improved with other params.
term_freq_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
X_train_tf = term_freq_transformer.transform(X_train_cv)
X_test_tf = term_freq_transformer.transform(X_test_cv)

# TODO: can be tested with other params, (alpha value)
multinominal_naive_bayes = MultinomialNB()
multinominal_naive_bayes.fit(X_train_tf, y_train)
nb_pred = multinominal_naive_bayes.predict(X_test_tf)

print(
    'Algorithm : Naive Bayes\n',
    classification_report(
        y_test, # from train and test split
        nb_pred, 
        target_names = sentiments # religion, age, ethnicity, gender, not_bullying
    )
)
print('\nCalculation completed...\n')
print(f'Total time: {round(time.time()-start_time, 2)}s')