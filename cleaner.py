import re
import string
import emoji
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# stop_words = [..., 'can', 'did', 'am', 'is', 'are', ...]
stop_words = set(stopwords.words('english'))


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
    text = clear_punctuation(text)
    text = clear_with_regex(text)
    text = clean_hashtags(text)
    text = clear_special_chars(text)
    text = stemmer(text)
    return text

def start_cleaning(df):
    texts_new = []
    for _tweet in df.text:
        texts_new.append(clean_tweets(_tweet))
    return texts_new