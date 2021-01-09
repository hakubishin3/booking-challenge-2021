import torch
from torch import nn


class BookingLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=500,
        rnn_dim=500,
        num_layers=2,
        dropout=0.3,
        rnn_dropout=0.3,
        tie_weight=False,
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=rnn_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
        )
        self.dense = nn.Linear(rnn_dim, vocab_size)

        if tie_weight:
            self.dense.weight = self.embedding.weight

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.tie_weight = tie_weight

    def forward(self, x_seq, h0=None):
        out_s, hidden = self.rnn(self.drop(self.embedding(x_seq)), h0)
        out_s = self.dense(
            self.drop(out_s.view(out_s.size(0) * out_s.size(1), out_s.size(2)))
        )
        return out_s, hidden
