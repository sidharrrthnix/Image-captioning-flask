import sys
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from model import EncoderCNN, DecoderRNN

import os
import torch
import json

import sys
from PIL import Image
from torchtext.data import Field, BucketIterator

from data_loader import get_loader
from vocabulary import Vocabulary
from translator import translate_sentence
from torchvision import transforms
from translator import Encoder,Decoder,Seq2Seq,get_translated_sentence

import en_core_web_sm , de_core_news_sm

from torchtext.datasets import TranslationDataset, Multi30k

spacy_en = en_core_web_sm.load()
spacy_de= de_core_news_sm.load()

CWD = os.getcwd()
print('Current Working Directory : '+CWD)

transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])


data_loader = get_loader(transform=transform_test,mode='test')
print(data_loader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



import os
import torch
from model import EncoderCNN, DecoderRNN


encoder_file = 'encoder-24.pkl'
decoder_file = 'decoder-24.pkl'

embed_size = 512
hidden_size = 512

vocab_size = len(data_loader.dataset.vocab)


encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

encoder.load_state_dict(torch.load(CWD+'/static/models/'+encoder_file))
decoder.load_state_dict(torch.load(CWD+'/static/models/'+decoder_file))


encoder.to(device)
decoder.to(device)



def clean_sentence(output):
    parts = [data_loader.dataset.vocab.idx2word[i] for i in output][1:-1]
    sentence = " ".join(parts)
    return sentence



def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

SRC = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

TRG = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True) 

train_data, valid_data, test_data = Multi30k.splits(exts = ('.en','.de'), 
                                                    fields = (SRC, TRG))            

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)              

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 1024
DEC_PF_DIM = 1024
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]              

model1 = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.load_state_dict(torch.load(CWD+'/static/models/tut6-model.pt'))

predicted_sentence = []
def CNN_PREDICT(path):
    
    image = Image.open(path)
    plt.imshow(image)
    plt.title('Sample Image')
    plt.show()
    image = transform_test(image).unsqueeze(0)
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output)
    print(sentence)
    english_translation = sentence
    predicted_sentence.clear()
    sentence = sentence.split(' ')
    predicted_sentence.append(sentence)
    print(predicted_sentence)
    
    german_translation =  get_translated_sentence(predicted_sentence[0])
    print(german_translation)
    return english_translation,german_translation


