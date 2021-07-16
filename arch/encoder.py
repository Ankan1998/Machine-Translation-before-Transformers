import torch
import torch.nn as nn
from data_utils.data_preparation import DataCreator

class Encoder(nn.Module):
    def __init__(self, inp_dim, emb_dim, hid_dim, dropout=0.2):
        super().__init__()

        self.hid_dim = hid_dim
        #self.n_layers = n_layers
        # inp_dim = no. of unique token in vocab [but arranged indexwise]
        # if tensor is tried with random number then inp_dim must be greater than the biggest number
        self.embedding = nn.Embedding(inp_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, bidirectional=True, dropout=dropout)
        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.embedding(src)
        #print(emb)
        embed_dropout = self.dropout(emb)
        out, (hidden, cell) = self.rnn(embed_dropout)
        hidden = torch.tanh(self.fc_h(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        cell = torch.tanh(self.fc_c(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1)))
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)


        return hidden, cell


if __name__ == "__main__":
    pass
    # data_c = DataCreator()
    # train_data, val_data, test_data = data_c.data_loader('data_utils/.data')
    # source, target = data_c.vocab_builder(train_data)
    # input_size_encoder = len(source.vocab)
    # encoder_embedding_size = 300
    # hidden_size = 1024
    # num_layers = 2
    # encoder_dropout = float(0.5)
    #
    # encoder_lstm = Encoder(input_size_encoder, hidden_size, encoder_embedding_size,
    #                             num_layers, encoder_dropout)
    # print(encoder_lstm)



