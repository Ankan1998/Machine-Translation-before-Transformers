import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, emb_dim, n_layers=2, dropout=0.2):
        super().__init__()
        # inp_dim = no. of unique token in vocab [but arranged indexwise]
        # if tensor is tried with random number then inp_dim must be greater than the biggest number
        self.embedding = nn.Embedding(inp_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.embedding(src)
        print(emb)
        embed_dropout = self.dropout(emb)
        out , (hidden, cell) = self.rnn(embed_dropout)

        return out, hidden, cell


if __name__ == "__main__":
    enc = Encoder(101, 40, 30, 4)
    inp = torch.LongTensor([[19.0,100,44.4]])
    out, h, c = enc(inp)
    print(out.shape)
    print(h.shape)
    print(c.shape)




