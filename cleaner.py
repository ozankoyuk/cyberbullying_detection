import re
import string
import emoji
import pandas as pd
import numpy as np
import time

from collections import Counter
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# stop_words = [..., 'can', 'did', 'am', 'is', 'are', ...]
stop_words = set(stopwords.words('english'))

def get_cyberbully_prob(prediction_results):
    religion_count = (prediction_results == 0).sum()
    age_count = (prediction_results == 1).sum()
    ethnicity_count = (prediction_results == 2).sum()
    gender_count = (prediction_results == 3).sum()
    not_cyberbullying_count = (prediction_results == 4).sum()
    cyberbullying_count = len(prediction_results) - not_cyberbullying_count
    cyberbully_percentage = (100 * cyberbullying_count) / len(prediction_results)

    return round(cyberbully_percentage, 2)

# Convert words into integer
def tokenize(column, seq_len):
    # add every word into corpus
    corpus = [word for text in column for word in text.split()]
    count_words = Counter(corpus)
    sorted_words = count_words.most_common()
    # sort most used to least used words and give numbers from 1 to N
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    # convert words in tweets into numbers that corresponding to the word.
    text_int = []
    for text in column:
        r = [vocab_to_int[word] for word in text.split()]
        text_int.append(r)

    # create empty matrix (tweet_count, max_tweet_length) = (38809, 79)
    features = np.zeros((len(text_int), seq_len), dtype = int)
    for i, review in enumerate(text_int):
        # check tweet length
        if len(review) <= seq_len:
            # append zeros to the head of tweet
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            # put only the max length
            new = review[: seq_len]
        # update matrix row like:
        # 0, 0, ..., 1, 2, 3
        features[i, :] = np.array(new)

    return sorted_words, features

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
    tokenized = word_tokenize(text)
    # TODO: maybe try SnowballStemmer()
    ps = PorterStemmer()
    return ' '.join([ps.stem(word) for word in tokenized])

# clear every tweet with this order
# TODO: can be improved later on
def clean_tweets(text):
    text = clear_with_regex(text)
    text = clear_punctuation(text)
    text = clean_hashtags(text)
    text = clear_special_chars(text)
    text = stemmer(text)
    return text

def start_cleaning(df):
    texts_new = []
    for _tweet in df.text:
        texts_new.append(clean_tweets(_tweet))
    return texts_new

def get_processed_df(max_tweet_length):
    start = time.time()

    df = pd.read_csv("cyberbullying_tweets.csv")
    df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})
    df.duplicated().sum()
    df = df[~df.duplicated()]
    df.sentiment.value_counts()

    texts_new = start_cleaning(df)

    df['text_clean'] = texts_new
    df["text_clean"].duplicated().sum()
    df.drop_duplicates("text_clean", inplace=True)
    df.sentiment.value_counts()
    df = df[df["sentiment"]!="other_cyberbullying"]

    text_len = []
    for text in df.text_clean:
        tweet_len = len(text.split())
        text_len.append(tweet_len)

    df['text_len'] = text_len

    df = df[df['text_len'] < max_tweet_length]

    df.sort_values(by=["text_len"], ascending=False)
    df['sentiment'] = df['sentiment'].replace({'religion':0,'age':1,'ethnicity':2,'gender':3,'not_cyberbullying':4})
    print(f"Elapsed time for pre-processing data:  {round(time.time()-start, 2)}s")
    
    return df