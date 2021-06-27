import torch
import torch.nn as nn
from data_utils.data_preparation import vocab_builder, data_loader


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=2, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, dropout=dropout)
        self.fc_out = nn.Linear(2 * hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell


if __name__ == "__main__":
    train_data, val_data, test_data = data_loader()
    source, target = vocab_builder(train_data)
    target_size_encoder = len(target.vocab)
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    dropout = float(0.5)

    decoder_lstm = Decoder(target_size_encoder, decoder_embedding_size,
                           hidden_size, num_layers, dropout)
    print(decoder_lstm)