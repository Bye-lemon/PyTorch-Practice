import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

WATCH = lambda x: print(x.shape)


class Dictionary(object):
    def __init__(self):
        super(Dictionary, self).__init__()
        self.word2idx = dict({})
        self.idx2word = []
        self.length = 0

    def __len__(self):
        return len(self.idx2word)

    def add_word(self, word):
        if word not in self.idx2word:
            self.idx2word.append(word)
            self.word2idx[word] = self.length
            self.length += 1
        return self.word2idx[word]

    def onehot_encoded(self, word):
        vec = np.zeros(self.length)
        vec[self.word2idx[word]] = 1
        return vec


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, target="lstm", bidirectional=False):
        super(RNN, self).__init__()
        self.net = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}.get(target)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = self.net(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x, state=None):
        hidden = state if state is not None else self._init_state(x)
        batch_size, sequence_length = x.size()
        x = self.embedding(x)
        x, _ = self.rnn(x, hidden)
        x = x.reshape(batch_size*sequence_length, -1)
        x = self.fc(x)
        return x, _

    def _init_state(self, x):
        if self.net is nn.RNN:
            return x.data.new(self.num_layers, x.shape[0], self.hidden_dim).fill_(0).float()
        elif self.net is nn.LSTM:
            return (
                x.data.new(self.num_layers, x.shape[0], self.hidden_dim).fill_(0).float(),
                x.data.new(self.num_layers, x.shape[0], self.hidden_dim).fill_(0).float()
            )


class LSTM(nn.Module):
    def __init__(self, sequence_length, embedding_dim, hidden_dim, output_dim, embedding, batch_size,
                 max_vocab_size, num_layers=1, dropout_rate=0.2, bidirection=False):
        super(LSTM, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = embedding
        self.batch_size = batch_size
        self.bidirection = bidirection
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_embeddings=max_vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.word_embedding(x)
        h_0 = self._init_state(batch_size=x.size(0))
        out, (h_t, c_t) = self.lstm(x, h_0)
        self.dropout(h_t)
        y_pred = self.fc(h_t[-1])
        y_pred = F.log_softmax(y_pred)
        return y_pred

    def _init_state(self, batch_size):
        weight = next(self.parameters()).data
        return (
            weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()
        )
