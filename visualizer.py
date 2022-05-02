#%%
import matplotlib.pyplot as plt
import wordcloud as wc
from sklearn.feature_extraction.text import CountVectorizer
import warnings

from plotly.offline import iplot
import pandas as pd
warnings.filterwarnings("ignore")

import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

from cleaner import get_processed_df

SENTIMENTS = {
    'Religion-based Cyberbullying':0,
    'Age-based Cyberbullying':1,
    'Ethnicity-based Cyberbullying':2,
    'Gender-based Cyberbullying':3,
    'Not Cyberbullying':4
}
MAX_TWEET_LENGTH = 100
CENSOR_LIST = {'fuck':'f_ck', 'nigger':'n__ger', 'nigga': 'n__ga', 'bitch': 'b__ch', 'ass': 'a_s'}

def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = wc.STOPWORDS).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

df = get_processed_df(MAX_TWEET_LENGTH)

for key, value in SENTIMENTS.items():
    data = df[df['sentiment'] == value]
    data_str = ""
    for txt in data['text_clean']:
        if any(ext in txt for ext in CENSOR_LIST.keys()):
            new_txt = txt
            for k, v in CENSOR_LIST.items():
                new_txt = new_txt.replace(k, v)
            data_str += ' ' + new_txt
        else:
            data_str += ' ' + txt

    wordcloud = wc.WordCloud(
        stopwords = wc.STOPWORDS,
        max_font_size = 80,
        width = 600,
        height = 600,
        background_color = 'black').generate(data_str)

    fig, ax = plt.subplots(figsize=(14,10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()
    plt.title(f'Wordcloud of {key}', size = 20)
    plt.imshow(wordcloud)

    unigrams = get_top_n_gram(data['text_clean'], (1,1), 10)
    bigrams = get_top_n_gram(data['text_clean'], (2,2), 10)

    gender_1 = pd.DataFrame(unigrams, columns = ['Text' , 'count'])
    gender_1.groupby('Text').\
        sum()['count'].\
        sort_values(ascending=True).\
        iplot(
            kind='bar', 
            xTitle='Count', 
            yTitle='Word', 
            linecolor='black',
            color='black', 
            title=f'Top 10 Unigrams for {key}',
            orientation='h'
        )

    gender_2 = pd.DataFrame(bigrams, columns = ['Text' , 'count'])
    gender_2.groupby('Text').\
        sum()['count'].\
        sort_values(ascending=True).\
        iplot(
            kind='bar', 
            yTitle='Word',
            xTitle='Count', 
            linecolor='black',
            color='black', 
            title=f'Top 10 Bigrams for {key}',
            orientation='h'
        )
