import torch
import torch.nn as nn
import torch.optim as optim
from seq2seq.model_init import model_initializer
from train import training
from data_utils.data_preparation import DataCreator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main_training(
        path='data_utils/.data',
        n_epochs = 50,
        clip = 1,
        model_saving_path = 'saved_model/seq2seq.pt',
        ENC_EMB_DIM = 256,
        DEC_EMB_DIM = 256,
        HID_DIM = 512,
        ENC_DROPOUT =0.5,
        DEC_DROPOUT = 0.5,
        DEVICE = DEVICE):

    data_c = DataCreator()
    train_data, val_data, test_data = data_c.data_loader(path)
    src, trg = data_c.vocab_builder(train_data)
    train_itr, val_itr, test_itr = data_c.data_iterator(train_data, val_data, test_data)

    INPUT_DIM = len(src.vocab)
    OUTPUT_DIM = len(trg.vocab)

    model = model_initializer(
        INPUT_DIM,
        OUTPUT_DIM,
        ENC_EMB_DIM,
        DEC_EMB_DIM,
        HID_DIM,
        ENC_DROPOUT,
        DEC_DROPOUT,
        DEVICE)

    TRG_PAD_IDX = trg.vocab.stoi[trg.pad_token]
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    training(model, train_itr, val_itr, optimizer, criterion,n_epochs,clip,model_saving_path)

if __name__ == "__main__":

    main_training()
