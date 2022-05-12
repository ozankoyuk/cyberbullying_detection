#%%
import pandas as pd
import numpy as np
import random
import time 
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from cleaner import get_processed_df, get_cyberbully_prob

seed_value = 20230337
random.seed(seed_value)
np.random.seed(seed_value)
warnings.filterwarnings(action="ignore")

def fit_model(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
    return accuracy, y_pred

MAX_TWEET_LENGTH = 100
df = get_processed_df(MAX_TWEET_LENGTH)

# split the data into test and train 
X = df['text_clean']
y = df['sentiment']

# train and test splitting: 
# %20 -> Test
# %80 -> Train
X_train, X_test, y_train, y_test = train_test_split(
    np.array(X), 
    y, 
    test_size = 0.2, 
    stratify = y, 
    random_state = seed_value
)

start = time.time()
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
frequency_matrix = TfidfVectorizer(
    use_idf = True, 
    tokenizer = word_tokenize,
    min_df = 0.00002,
    max_df = 0.70
)

# convert X_train and X_test into unicode
# and then to a matrix of TF-IDF features
X_train_tf = frequency_matrix.fit_transform(
    X_train.astype('U')
)
X_test_tf = frequency_matrix.transform(
    X_test.astype('U')
)

random_forest = RandomForestClassifier(
    random_state=42, 
    criterion='gini',
    min_samples_split=3,
    min_samples_leaf=2,
)

ada_boost = AdaBoostClassifier(
    random_state=42,
    learning_rate=0.0000005
)

# possible metrics : mlogloss, logloss, mae, mape, auc
# all metrics will give the same result in my case
xg_boost_mlogloss = XGBClassifier(
    eval_metric="mlogloss",
    random_state=42,
    eta=0.0000005,
    max_depth=10
)

decision_tree = DecisionTreeClassifier(
    random_state=42,
    min_samples_split=3,
    min_samples_leaf=2,
)

perceptron_algorithm = MLPClassifier(
    random_state=42,
    hidden_layer_sizes=(100,),
    batch_size=32,
)

print(f"Elapsed time for preparing models:  {round(time.time()-start, 2)}s")

algorithms = {
    "Random Forest": random_forest,
    "XGBoost (MLogLoss)": xg_boost_mlogloss,
    "Decision Tree": decision_tree,
    "AdaBoost Algorithm": ada_boost
    # multilayer perceptron takes too much time to perfom
    #"Multilayer Perceptron": perceptron_algorithm
}

accuracy_list = []
probability_list = []
time_list = []

for name, _model in algorithms.items():
    start = time.time()
    # fit train and test data
    curr_acc, predicted_list = fit_model(
        _model, 
        X_train_tf, 
        y_train, 
        X_test_tf, 
        y_test
    )
    
    # add accuracy, probability and the time values
    accuracy_list.append(curr_acc)
    probability_list.append(get_cyberbully_prob(predicted_list))
    time_list.append(
        round(time.time()-start, 2)
    )

# show every data in a table
models_df = pd.DataFrame(
    {
        "Algorithm" : algorithms.keys(),
        "Accuracy (%)" : accuracy_list,
        "Probability (%)" : probability_list,
        "Total Time (s) " : time_list
    }
).sort_values("Accuracy (%)", ascending=False)
print(models_df)

#
#  ___                   _  __                 _    
# / _ \ ______ _ _ __   | |/ /___  _   _ _   _| | __
#| | | |_  / _` | '_ \  | ' // _ \| | | | | | | |/ /
#| |_| |/ / (_| | | | | | . \ (_) | |_| | |_| |   < 
# \___//___\__,_|_| |_| |_|\_\___/ \__, |\__,_|_|\_\
#                                  |___/            
#