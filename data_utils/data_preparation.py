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

spacy_fr = spacy.load('fr_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_fr(text):
    # Reversing sentences to get long term benefit
    return [tok.text for tok in spacy_fr.tokenizer(text)][::-1]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


