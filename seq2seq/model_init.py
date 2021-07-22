import torch
import torch.nn as nn
from seq2seq.encoder import Encoder
from seq2seq.decoder import Decoder
from seq2seq.se2seq import Seq2seq


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def model_initializer(
        INPUT_DIM,
        OUTPUT_DIM,
        ENC_EMB_DIM,
        DEC_EMB_DIM,
        HID_DIM,
        ENC_DROPOUT,
        DEC_DROPOUT,
        DEVICE='cpu'):
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
    model = Seq2seq(enc, dec).to(DEVICE)
    model.apply(init_weights)

    return model
