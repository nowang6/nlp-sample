#https://www.kaggle.com/code/vikkach/sentiment-analysis-lstm-pytorch
import torch

from tqdm import tqdm
from torch import nn, optim
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import os

class Config():
    # dataset
    train_data_path = 'data\\train.tsv'
    test_data_path = 'data\\test.tsv'
    pad_inputs = 0
    # training
    batch_size = 12
    learning_rate = 0.001
    num_epochs = 100
    clip_value = 5
    eval_every = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model
    output_size = 1
    embedding_dim = 300
    hidden_dim = 256
    n_layers = 2
    n_classes = 5
    dropout = 0.5
    checkpoint = "siebert/sentiment-roberta-large-english"
    # testing
    test_batch_size = 4
    model_path = 'model\\SentimentRNN.pt'


train_data = pd.read_csv(Config.train_data_path, sep = '\t')
test_data = pd.read_csv(Config.test_data_path, sep = '\t')


def pre_process(df):
    reviews = []
    stopwords_set = set(stopwords.words("english"))
    ps = PorterStemmer()
    for p in tqdm(df['Phrase']):
        # convert to lowercase
        if (not isinstance(p, str)):
            p = ""
        p = p.lower()
        # remove punctuation and additional empty strings
        p = ''.join([c for c in p if c not in punctuation])
        reviews_split = p.split()
        reviews_wo_stopwords = [word for word in reviews_split if not word in stopwords_set]
        reviews_stemm = [ps.stem(w) for w in reviews_wo_stopwords]
        p = ' '.join(reviews_stemm)
        reviews.append(p)
    return reviews


train_data_pp = pre_process(train_data)
test_data_pp = pre_process(test_data)

# compare the same phrase before and after pre-processing
print('Phrase before pre-processing: ', train_data['Phrase'][0])
print('Phrase after pre-processing: ', train_data_pp[0])

def encode_words(data_pp):
    words = []
    for p in data_pp:
        words.extend(p.split())
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab_to_int

encoded_voc = encode_words(train_data_pp + test_data_pp)

def encode_data(data):
    reviews_ints = []
    for ph in data:
        reviews_ints.append([encoded_voc[word] for word in ph.split()]) 
    return reviews_ints

train_reviews_ints = encode_data(train_data_pp)
test_reviews_ints = encode_data(test_data_pp)
print('Example of encoded train data: ', train_reviews_ints[0])
print('Example of encoded test data: ', test_reviews_ints[0])

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

y_target = to_categorical(train_data['Sentiment'], Config.n_classes)
print('Example of target: ', y_target[0])

train_review_lens = Counter([len(x) for x in train_reviews_ints])
print("Zero-length train reviews: ", train_review_lens)
print("Maximum train review length: ", max(train_review_lens))


non_zero_idx = [ii for ii, review in enumerate(train_reviews_ints) if len(review) != 0]

train_reviews_ints = [train_reviews_ints[ii] for ii in non_zero_idx]
y_target = np.array([y_target[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers: ', len(train_reviews_ints))

def pad_features(reviews, seq_length):
    features = np.zeros((len(reviews), seq_length), dtype=int)

    for i, row in enumerate(reviews):
        try:
            features[i, -len(row):] = np.array(row)[:seq_length]
        except ValueError:
            continue
    
    return features

train_features = pad_features(train_reviews_ints, max(train_review_lens))
X_test = pad_features(test_reviews_ints, max(train_review_lens))

X_train,X_val,y_train,y_val = train_test_split(train_features,y_target,test_size = 0.2)
print(X_train[0])
print(y_train[0])
print("X_train",X_train.shape)
print("X_val",X_val.shape)
print("X_test",X_test.shape)
train_size = X_train.shape[0] - X_train.shape[0] % Config.batch_size
val_size = X_val.shape[0] - X_val.shape[0] % Config.batch_size
X_train = X_train[:train_size]
X_val = X_val[:val_size]
y_train = y_train[:train_size]
y_val = y_val[:val_size]

ids_test = np.array([t['PhraseId'] for ii, t in test_data.iterrows()])


train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(ids_test))

train_loader = DataLoader(train_data, shuffle=True, batch_size=Config.batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=Config.batch_size)
test_loader = DataLoader(test_data, batch_size=Config.test_batch_size)

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob):
        """
        Initialize the model by setting up the layers.
        Arguments:
        vocab_size - The size of the vocabulary, i.e., the total number of unique words in the input data.
        output_size - The size of the output, which is usually set to 1 for binary classification tasks like sentiment analysis.
        embedding_dim - The dimensionality of the word embeddings. Each word in the input data will be represented by a dense vector of this dimension.
        hidden_dim - The number of units in the hidden state of the LSTM layer.
        n_layers - The number of layers in the LSTM.
        drop_prob - The probability of dropout, which is a regularization technique used to prevent overfitting.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # an embedding layer that maps each word index to its dense vector representation. 
        # this layer is used to learn word embeddings during training.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # an LSTM layer that processes the input sequence of word embeddings 
        # and produces a sequence of hidden states. 
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            n_layers, 
                            dropout=0.5, 
                            batch_first=True)
        
        # a dropout layer that randomly sets elements of the input to zero 
        # with probability drop_prob. 
        # this layer helps in preventing overfitting.
        self.dropout = nn.Dropout(p=drop_prob)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        # compute the word embeddings for the input sequence.
        batch_size = x.size(0)
        embeds = self.embedding(x)
        
        # pass the embeddings through the LSTM layer to get the LSTM outputs and the updated hidden state.
        lstm_out, hidden = self.lstm(embeds,hidden)
        lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)
        
        # apply dropout to the reshaped LSTM outputs.
        out = self.dropout(lstm_out)
        
        # pass the output through the fully connected layer.
        out = self.fc(out)
        
        # apply the sigmoid activation function to squash the output between 0 and 1.
        out = self.sig(out)
        out = out.view(batch_size,-1)
        
        # extract the last five elements from each sequence in the batch
        out = out[:,-5:]
        return out, hidden
    
    
    def init_hidden(self, batch_size, device):
        """ 
        Initializes hidden state 
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        
        return hidden


def train_loop(model, optimizer, criterion, train_loader, clip_value, device, batch_size=Config.batch_size):
    """
    Train model
    """
    running_loss = 0
    model.train()
    
    # returns the initial hidden state for the LSTM layers
    h = model.init_hidden(batch_size, device)
    
    for seq, targets in train_loader:
        # move data to device
        seq = seq.to(device)
        targets = targets.to(device)
        
        # convert the elements of the hidden state tuple h to tensors with the same device as the input data.
        h = tuple([each.data for each in h])
        
        # perform a forward pass through the model.
        # returns the model's output (out) and the updated hidden state (h).
        out, h = model.forward(seq, h)
        
        # calculate the loss between the predicted output and the target values
        loss = criterion(out, targets.float())
        running_loss += loss.item()*seq.shape[0]
        
        # reset the gradients of the model's parameters 
        optimizer.zero_grad()
        
        # compute the gradients of the loss with respect to the model's parameters
        loss.backward()
        if clip_value:
            
            # clip the gradients to prevent them from exploding
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
        # update the model's parameters
        optimizer.step()
    running_loss /= len(train_loader.sampler)
    return running_loss

def get_prediction(t):
    max_indices = torch.argmax(t, dim=1)
    new = torch.zeros_like(t)
    new[torch.arange(t.shape[0]), max_indices] = 1
    return new


def eval_loop(model, criterion, eval_loader, device, batch_size=Config.batch_size, ignore_index=None):
    """
    Evaluate model
    """
    
    # returns the initial hidden state for the LSTM layers
    val_h = model.init_hidden(batch_size, device)
    val_loss = 0
    model.eval()
    accuracy = []
    for seq, targets in eval_loader:
        
        # convert the elements of the hidden state tuple val_h to tensors with the same device as the input data.
        val_h = tuple([each.data for each in val_h])
        
        # move data to device
        seq = seq.to(device)
        targets = targets.to(device)

        # perform a forward pass through the model.
        # returns the model's output (out) and the updated hidden state (val_h).
        out, val_h = model(seq, val_h)
        
        # calculate the loss
        loss = criterion(out, targets.float())
        val_loss += loss.item()*seq.shape[0]
        
        # convert the model's output
        predicted = get_prediction(out).flatten().cpu().numpy()
        labels = targets.view(-1).cpu().numpy()
        
        # calculate the accuracy score between the predicted and target values
        accuracy.append(accuracy_score(labels, predicted))
        
    acc = sum(accuracy)/len(accuracy)
    val_loss /= len(eval_loader.sampler)
    return {'accuracy':acc,
            'loss':val_loss}

def train(model, optimizer, criterion, train_loader, valid_loader,
          eval_every, num_epochs, clip_value, 
          ignore_index=None,
          device=Config.device,  
          valid_loss_min=np.inf):
    
    for e in range(num_epochs):
        # train for epoch
        train_loss = train_loop(model, optimizer, criterion, train_loader, clip_value, device)
        
        if (e+1) % eval_every == 0:

            # evaluate on validation set
            metrics = eval_loop(model, criterion, valid_loader, device)

            # show progress
            print_string = f'Epoch: {e+1} '
            print_string+= f'TrainLoss: {train_loss:.5f} '
            print_string+= f'ValidLoss: {metrics["loss"]:.5f} '
            print_string+= f'ACC: {metrics["accuracy"]:.5f} '
            print(print_string)

            # save the model 
            if metrics["loss"] <= valid_loss_min:
                torch.save(model.state_dict(), Config.model_path)
                valid_loss_min = metrics["loss"]

vocab_size = len(encoded_voc)+1 # +1 for the 0 padding + our word tokens
model = SentimentRNN(vocab_size, Config.output_size, Config.embedding_dim, Config.hidden_dim, Config.n_layers, Config.dropout)
model = model.to(Config.device)
optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
criterion = nn.BCELoss()
print(os.getpid())
train(model, optimizer, criterion, train_loader, valid_loader,
      Config.eval_every, Config.num_epochs, Config.clip_value)
