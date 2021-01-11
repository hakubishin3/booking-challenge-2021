import torch
from torch import nn


class BookingLSTM(nn.Module):
    def __init__(
        self,
        n_city_id,
        n_booker_country,
        n_device_class,
        n_affiliate_id,
        n_month_checkin,
        n_num_checkin,
        emb_dim=512,
        rnn_dim=512,
        num_layers=2,
        dropout=0.3,
        rnn_dropout=0.3,
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.city_id_embedding = nn.Embedding(n_city_id, emb_dim)
        self.booker_country_embedding = nn.Embedding(n_booker_country, emb_dim)
        self.device_class_embedding = nn.Embedding(n_device_class, emb_dim)
        self.affiliate_id_embedding = nn.Embedding(n_affiliate_id, emb_dim)
        self.month_checkin_embedding = nn.Embedding(n_month_checkin, emb_dim)
        self.num_checkin_embedding = nn.Embedding(n_num_checkin, emb_dim)

        self.linear = nn.Linear(emb_dim * 6 + 1, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=rnn_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(rnn_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, n_city_id),
        )

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
        device_class_tensor,
        affiliate_id_tensor,
        month_checkin_tensor,
        num_checkin_tensor,
        days_stay_tensor,
    ):
        city_id_embedding = self.city_id_embedding(city_id_tensor)
        booker_country_embedding = self.booker_country_embedding(booker_country_tensor)
        device_class_embedding = self.device_class_embedding(device_class_tensor)
        affiliate_id_embedding = self.affiliate_id_embedding(affiliate_id_tensor)
        month_checkin_embedding = self.month_checkin_embedding(month_checkin_tensor)
        num_checkin_embedding = self.num_checkin_embedding(num_checkin_tensor)
        days_stay_feature = days_stay_tensor.unsqueeze(2)

        out_s = torch.cat(
            [
                city_id_embedding,
                booker_country_embedding,
                device_class_embedding,
                affiliate_id_embedding,
                month_checkin_embedding,
                num_checkin_embedding,
            ],
            dim=2,
        )
        out_s = torch.cat([out_s, days_stay_feature], dim=2)

        out_s = self.linear(self.drop(out_s))
        out_s, _ = self.lstm(out_s)
        out_s = self.ffn(
            out_s.contiguous().view(out_s.size(0) * out_s.size(1), out_s.size(2))
        )
        return out_s
