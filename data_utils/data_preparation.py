import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

SEED = 9999
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    # Reversing sentences to get long term benefit
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


src = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

trg = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)


train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(src, trg))

# print(f"Number of training examples: {len(train_data.examples)}")
# print(f"Number of validation examples: {len(valid_data.examples)}")
# print(f"Number of testing examples: {len(test_data.examples)}")

src.build_vocab(train_data, min_freq = 2)
trg.build_vocab(train_data, min_freq = 2)

print(f"Unique tokens in source (de) vocabulary: {len(src.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(trg.vocab)}")
