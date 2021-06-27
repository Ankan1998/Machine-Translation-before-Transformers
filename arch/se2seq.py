import random
import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from data_utils.data_preparation import vocab_builder, data_loader
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, tfr=0.5):
        # 1st argument timestep(length of sequence of line
        len_seq_trg = trg.shape[0]
        batch_size = trg.shape[1]
        outputs = torch.zeros(len_seq_trg, batch_size, len(trg.vocab))
        hidden, cell = self.encoder(src)
        X = trg[0]
        for i in range(1, len_seq_trg):
            output, hidden_state, cell_state = self.decoder(X, hidden, cell)
            outputs[i] = output
            res = output.argmax(1)  # 1st word Embedding
            X = trg[i] if random.random() < tfr else res

        return outputs


if __name__ == "__main__":
    train_data, val_data, test_data = data_loader()
    source, target = vocab_builder(train_data)
    input_size_encoder = len(source.vocab)
    target_size_encoder = len(target.vocab)
    embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    dropout = float(0.5)

    encoder_lstm = Encoder(input_size_encoder, hidden_size, embedding_size,
                                num_layers, dropout)
    decoder_lstm = Decoder(target_size_encoder, embedding_size,
                               hidden_size, num_layers, dropout)
    s2s = Seq2seq(encoder_lstm, decoder_lstm)
    print(s2s)
