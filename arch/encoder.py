import torch
import torch.nn as nn
from data_utils.data_preparation import vocab_builder, data_loader

class Encoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, emb_dim, n_layers=2, dropout=0.2):
        super().__init__()
        # inp_dim = no. of unique token in vocab [but arranged indexwise]
        # if tensor is tried with random number then inp_dim must be greater than the biggest number
        self.embedding = nn.Embedding(inp_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.embedding(src)
        print(emb)
        embed_dropout = self.dropout(emb)
        out , (hidden, cell) = self.rnn(embed_dropout)

        return out, hidden, cell


if __name__ == "__main__":
    # enc = Encoder(101, 40, 30, 4)
    # inp = torch.LongTensor([[19.0,100,44.4]])
    # out, h, c = enc(inp)
    # print(out.shape)
    # print(h.shape)
    # print(c.shape)
    train_data, val_data, test_data = data_loader()
    source, target = vocab_builder(train_data)
    input_size_encoder = len(source.vocab)
    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = float(0.5)

    encoder_lstm = Encoder(input_size_encoder, hidden_size, encoder_embedding_size,
                                num_layers, encoder_dropout)
    print(encoder_lstm)



