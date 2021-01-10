import torch
from torch import nn


class BookingLSTM(nn.Module):
    def __init__(
        self,
        n_city_id=39902,
        n_booker_country=5,
        emb_dim=500,
        rnn_dim=500,
        num_layers=2,
        dropout=0.3,
        rnn_dropout=0.3,
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.city_id_embedding = nn.Embedding(n_city_id, emb_dim)
        self.booker_country_embedding = nn.Embedding(n_booker_country, emb_dim)
        self.rnn = nn.RNN(
            input_size=emb_dim * 2,
            hidden_size=rnn_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=False,
        )
        self.dense = nn.Linear(rnn_dim, n_city_id)

        self.n_city_id = n_city_id
        self.n_booker_country = n_booker_country
        self.emb_dim = emb_dim
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout

    def forward(
        self,
        city_id_tensor,
        booker_country_tensor,
        h0=None):
        city_id_embedding = self.city_id_embedding(city_id_tensor)
        booker_country_embedding = self.booker_country_embedding(booker_country_tensor)

        out_s = torch.cat([city_id_embedding, booker_country_embedding], dim=2)
        out_s, hidden = self.rnn(self.drop(out_s), h0)
        out_s = self.dense(
            self.drop(out_s.view(out_s.size(0) * out_s.size(1), out_s.size(2)))
        )
        return out_s, hidden
