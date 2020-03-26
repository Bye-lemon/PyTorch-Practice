import torch
import torch.nn as nn
import torch.nn.functional as F


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

