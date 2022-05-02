#%%
import numpy as np
import random
import time
import warnings
import torch
import torch.nn as nn

from cleaner import get_processed_df, tokenize, get_cyberbully_prob
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

seed_value=42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

MAX_TWEET_LENGTH = 100
# size of data used in every iteration
BATCH_SIZE = 32
# number of neurons of the internal neural network in the LSTM
HIDDEN_DIM = 100 
# stacked LSTM layers
LSTM_LAYERS = 1 
# learning rate of optimizer changes over time:
LR = 3e-4 
# LSTM Dropout
DROPOUT = 0.5 
# number of training epoch/iteration
EPOCHS = 1
BIDIRECTIONAL = True
DIRECTION_COUNT = 2
EMBEDDING_DIM = 50
SENTIMENTS = ["religion", "age", "ethnicity", "gender", "not bullying"]
NUM_CLASSES = len(SENTIMENTS)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm_layers = LSTM_LAYERS
        self.hidden_dim = HIDDEN_DIM
        self.num_classes = NUM_CLASSES
        self.batch_size = BATCH_SIZE
        
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        
        self.lstm = nn.LSTM(EMBEDDING_DIM,
                            HIDDEN_DIM,
                            num_layers=LSTM_LAYERS,
                            dropout=DROPOUT,
                            bidirectional=BIDIRECTIONAL,
                            batch_first=True)

        self.connect_layers = nn.Linear(HIDDEN_DIM * DIRECTION_COUNT, NUM_CLASSES)
        self.log_soft_max = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # in this function, we need to re-initialize model
        # within the order in the __init__function : 
        # embedding -> lstm -> layer connection -> logsoftmax

        # set new batch size
        self.batch_size = x.size(0)

        # create embedding
        embedded = self.embedding(x)
        
        # create model and hidden state
        model, hidden_state = self.lstm(embedded, hidden)
        
        # get hidden state from the last LSTM cell
        model = model[:,-1,:]
        
        # connect layers
        model = self.connect_layers(model)

        # set soft max
        model = self.log_soft_max(model)

        return model, hidden_state

def create_hidden_layer(labels):
    hidden_state = torch.zeros(
        (
            LSTM_LAYERS * DIRECTION_COUNT, 
            labels.size(0), 
            HIDDEN_DIM
        )
    ).detach().to('cpu')
    cell_state = torch.zeros(
        (
            LSTM_LAYERS * DIRECTION_COUNT, 
            labels.size(0), 
            HIDDEN_DIM
        )
    ).detach().to('cpu')
    return (hidden_state, cell_state)

start = time.time()

df = get_processed_df(MAX_TWEET_LENGTH)
max_len = np.max(df['text_len'])

X = df['text_clean']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_value)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=seed_value)

print(f"Elapsed time in seconds for pre-processing:  {round(time.time()-start, 2)}s")

start = time.time()

vocabulary, tokenized_column = tokenize(df["text_clean"], max_len)
VOCAB_SIZE = len(vocabulary) + 1 

# convert tweets' words into lists
Word2vec_train_data = list(map(lambda x: x.split(), X_train))
# create vectors for words in tweets
word2vec_model = Word2Vec(
    Word2vec_train_data, 
    vector_size=EMBEDDING_DIM,
    epochs=10,
    window=8
    )

# define empty embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    
# fill matrix with the data of word2vec
# every row corresponds to the vector of that word from word2vec
for word, token in vocabulary:
    if word2vec_model.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model.wv.__getitem__(word)

# at this point, we have:
# X -> tweets that converted to numbers (words)
# y -> tweet sentiments
X = tokenized_column
y = df['sentiment'].values

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

# I need to balance the classes
# to handle this, I use  RandomOverSampler from imblearn library
# after this, every category will have the same amount of data inside of them
ros = RandomOverSampler()
X_train_os, y_train_os = ros.fit_resample(
    np.array(X_train),
    np.array(y_train)
)

print(f"Elapsed time in seconds to tokenize tweets:  {round(time.time()-start, 2)}s")

start = time.time()

train_data = TensorDataset(
    torch.from_numpy(X_train_os), 
    torch.from_numpy(y_train_os)
)
test_data = TensorDataset(
    torch.from_numpy(X_test), 
    torch.from_numpy(y_test)
)
valid_data = TensorDataset(
    torch.from_numpy(X_valid), 
    torch.from_numpy(y_valid)
)

# DataLoader contains datasets and datasets contains tensors
# these tensors are X_train_os and y_train_os
# params: 
#   shuffle     : shuffle data in every epoch. in every iteration, 
#                 random data from the TensorDataset will be consumed
#   batch_size  : create batch to iterate them, when shuffle enabled 
#                 every data will be random. 
#                 if it is false, then data will be used in order
#   drop_last   : when there is no enough data to create a batch, drop it
train_loader = DataLoader(
    train_data, 
    shuffle=True, 
    batch_size=BATCH_SIZE, 
    drop_last=True)
valid_loader = DataLoader(
    valid_data, 
    shuffle=True, 
    batch_size=BATCH_SIZE, 
    drop_last=True)
test_loader = DataLoader(
    test_data, 
    shuffle=True, 
    batch_size=BATCH_SIZE, 
    drop_last=True)

print(f"Elapsed time in seconds to prepare DataLoaders:  {round(time.time()-start, 2)}s")

start = time.time()

lstm_model = LSTM()
lstm_model = lstm_model.to('cpu')
"""
[
    ('embedding', Embedding(33286, 300)),
    ('lstm', LSTM(300, 100, batch_first=True, dropout=0.5, bidirectional=True)),
    ('fc', Linear(in_features=200, out_features=5, bias=True)),
    ('softmax', LogSoftmax(dim=1))
]
"""

# clear embedding weight for train
lstm_model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

approach = nn.LogSoftmax(dim=1)
criterion = nn.NLLLoss()

optimizer_AdamW = torch.optim.AdamW(
    lstm_model.parameters(), 
    lr = LR, 
    weight_decay = 5e-6
)
# optimizer_Adamax = torch.optim.Adamax(
#   lstm_model.parameters(),
#   lr = LR, 
#   weight_decay = 5e-6
# )
# optimizer_Adagrad = torch.optim.Adagrad(
#   lstm_model.parameters(),
#   lr = LR, 
#   weight_decay = 5e-6
# )
# optimizer_Adadelta = torch.optim.Adadelta(
#   lstm_model.parameters(),
#   lr=LR,
#   weight_decay = 5e-6
# )

total_step = len(train_loader)
total_step_val = len(valid_loader)
best_accuracy = 0
all_accuracy_percentage = []
best_state ={}

print(f"Elapsed time in seconds to initialize model:  {round(time.time()-start, 2)}s")

for e in range(EPOCHS):
    start = time.time()

    # save changes of every batch for each iteration
    validation_accuracy = []
    all_predictions_list = []
    y_val_list = []

    # count correctly_predicted_count classified tweets
    correctly_predicted_count = 0
    correct_count_from_validation = 0
    total_predicted_count = 0
    total_predicted_from_validation = 0

    lstm_model.train()

    # train_loader contains tensors with X_train_os and y_train_os
    for inputs, labels in train_loader:
        # get batch sized tensor data and classes.
        # every input.shape will be like this: [BATCH_SIZE, VOCAB_MAX_LEN]
        # to increase the speed of the predictions, 
        # we need to load data into the cpu
        inputs = inputs.to('cpu')
        labels = labels.to('cpu')

        # initialization of the LSTM hidden states
        # LSTM_LAYERS * DIRECTION_COUNT -> Bidirectional LSTM.
        hidden_layer = create_hidden_layer(labels)

        # we need to call zero_grad() because after gradients are computed in backward(),
        # we need to use step() to proceed gradiend descent. however gradients are not
        # automatically zeroed due to these two functions.
        # if we don't set zero the gradients, the results will be inaccurate and wrong.
        lstm_model.zero_grad()

        # generate output from hidden layer
        epoch_output, hidden_layer = lstm_model(inputs, hidden_layer)
        
        # recall last output of the network
        nll_Loss = criterion(approach(epoch_output), labels)
        nll_Loss.backward()
                
        optimizer_AdamW.step()

        # array of predicted values from training set size of BATCH_SIZE
        predicted_from_train = torch.argmax(epoch_output, dim=1)
        
        # save all predicted values
        all_predictions_list.extend(
            predicted_from_train.squeeze().tolist()
        )
        
        # total_predicted_count number of correctly predicted values.
        correctly_predicted_count += torch.sum(
            predicted_from_train == labels
        ).item()
        
        # sum of values in the size of BATCH_SIZE
        total_predicted_count += labels.size(0) 

   
    # when backward is not necessary, we can disable gradient calculation
    # to reduce memory consumption for computations
    with torch.no_grad():
        # enable evaluation mode and train
        lstm_model.eval()
        
        for inputs, labels in valid_loader:
            # to increase the speed of the predictions, 
            # we need to load data into the cpu
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')
            
            validation_hidden_layer = create_hidden_layer(labels)

            validation_output, validation_hidden_layer = lstm_model(
                inputs, validation_hidden_layer
            )

            predicted_from_validation = torch.argmax(validation_output, dim=1)
            y_val_list.extend(
                predicted_from_validation.squeeze().tolist()
            )

            correct_count_from_validation += torch.sum(
                predicted_from_validation == labels
            ).item()
            total_predicted_from_validation += labels.size(0)

        validation_accuracy.append(
            100 * correct_count_from_validation / total_predicted_from_validation
        )

    # find if accuracy increased or not
    if np.mean(validation_accuracy) >= best_accuracy:
        best_state = lstm_model.state_dict()
        best_accuracy = np.mean(validation_accuracy)
    
    all_accuracy_percentage.append(np.mean(validation_accuracy))
    print(f"Elapsed time in seconds to complete EPOCH_{e}:  {round(time.time()-start, 2)}s")

start = time.time()

# reload the state into model
lstm_model.load_state_dict(best_state)
lstm_model.eval()

predicted_list = []
test_list = []

for inputs, labels in test_loader:
    # load data from loader into the cpu
    inputs = inputs.to('cpu')
    labels = labels.to('cpu')

    test_hidden_layer = create_hidden_layer(labels)

    model_output, hidden_values = lstm_model(inputs, test_hidden_layer)
    # argmax with dimension returns the index of max value in the data
    # ex = [[1,2,3], [3,2,1], [1,1,2]]
    # argmax(ex, dim=1) -> [2, 0, 2]
    predicted_test = torch.argmax(model_output, dim=1)

    predicted_list.extend(predicted_test.squeeze().tolist())
    test_list.extend(labels.squeeze().tolist())

accuracy_percentage = classification_report(
    test_list, 
    predicted_list, 
    target_names = SENTIMENTS,
    output_dict = True
)['accuracy']*100

print(f"Elapsed time in seconds to retrieve results:  {round(time.time()-start, 2)}s")
print("Train accuracy for every epoch", all_accuracy_percentage)
print(
    'Classification Report for Bi-LSTM :\n', 
    classification_report(test_list, predicted_list, target_names=SENTIMENTS)
)
print(
    f"Considering the tweets of the user, "
    f"it was decided that this user is a cyberbully with:\n"
    f"Probability\t{get_cyberbully_prob(np.array(predicted_list))}%\n"
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