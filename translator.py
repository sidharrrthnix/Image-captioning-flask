import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
from data_loader import get_loader
import math
import torch.utils.data as data
import os
from vocabulary import Vocabulary
import json
from PIL import Image
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset, Multi30k
import time
from model import EncoderCNN, DecoderRNN

CWD = os.getcwd()
print('Current Working Directory : '+CWD)

batch_size = 128  
vocab_threshold = 4  
vocab_from_file = True  
embed_size = 512  
hidden_size = 512  
num_epochs = 20  
save_every = 1  
print_every = 100  
log_file = 'training_log.txt'  





transform_train = transforms.Compose([
    transforms.Resize(256),  
    transforms.RandomCrop(224),  
    transforms.RandomHorizontalFlip(
    ),  
    transforms.ToTensor(),  
    transforms.Normalize(
        (0.485, 0.456, 0.406),  
        (0.229, 0.224, 0.225))
])

transform_val = transforms.Compose([
    transforms.Resize(256), 
    transforms.RandomCrop(224), 
    transforms.ToTensor(),  
    transforms.Normalize(
        (0.485, 0.456, 0.406),  
        (0.229, 0.224, 0.225))
])




class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

def weights_init(m):

    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first = True, dropout = 0.5, num_layers = self.num_layers)
    
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        
        self = self.apply(weights_init)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        batch_size = features.size(0)
        
        self.hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        self.cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)

        captions_embed = self.embed(captions)
        
        vals = torch.cat((features.unsqueeze(1), captions_embed), dim=1)
        outputs, (self.hidden_state, self.cell_state) = self.lstm(vals, (self.hidden_state, self.cell_state))
        
        outputs = self.fc_out(outputs)
            
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        output = []
        batch_size = inputs.shape[0]
        
        self.hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        self.cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
    
        while True: 
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(inputs, (self.hidden_state, self.cell_state))
            
            outputs = self.fc_out(lstm_out)
            
            outputs = outputs.squeeze(1)
            _, max_indice = torch.max(outputs, dim=1) 
            
            output.append(max_indice.cpu().numpy()[0].item()) 
            
            if (max_indice == 1 or len(output) >= max_len):
                break
            
            inputs = self.embed(max_indice) 
            inputs = inputs.unsqueeze(1)
            
        return output




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available(
) else nn.CrossEntropyLoss()





transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





import os
import torch
from model import EncoderCNN, DecoderRNN

encoder_file = 'encoder-24.pkl'
decoder_file = 'decoder-24.pkl'


embed_size = 512
hidden_size = 512


encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, 9955)
decoder.eval()

encoder.load_state_dict(torch.load(CWD+'/static/models/'+encoder_file))
decoder.load_state_dict(torch.load(CWD+'/static/models/'+decoder_file))



encoder.to(device)
decoder.to(device)


def clean_sentence(output):
    sentence = ''
    for i in output:
        word = data_loader.dataset.vocab.idx2word[i]
        if i == 0:
            continue
        if i == 1:
            break
        if i == 18:
            sentence = sentence + word
        else:
            sentence = sentence + ' ' + word

    return sentence.strip()
    return sentence



import en_core_web_sm , de_core_news_sm

spacy_en = en_core_web_sm.load()
spacy_de= de_core_news_sm.load()




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





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




BATCH_SIZE = 16

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)




class Encoder(nn.Module):
   def __init__(self, 
                input_dim, 
                hid_dim, 
                n_layers, 
                n_heads, 
                pf_dim,
                dropout, 
                device,
                max_length = 100):
       super().__init__()

       self.device = device
       
       self.tok_embedding = nn.Embedding(input_dim, hid_dim)
       self.pos_embedding = nn.Embedding(max_length, hid_dim)
       
       self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                 n_heads, 
                                                 pf_dim,
                                                 dropout, 
                                                 device) 
                                    for _ in range(n_layers)])
       
       self.dropout = nn.Dropout(dropout)
       
       self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
       
   def forward(self, src, src_mask):
       
      
       
       batch_size = src.shape[0]
       src_len = src.shape[1]
       
       pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
       
       
       
       src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
       
       
       
       for layer in self.layers:
           src = layer(src, src_mask)
           
       
           
       return src



class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        
                
        
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
       
        
        
        _src = self.positionwise_feedforward(src)
        
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        
        return src




class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
       
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        
                
        x = torch.matmul(self.dropout(attention), V)
        
        
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        
        
        x = self.fc_o(x)
        
        
        
        return x, attention





class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        
        x = self.fc_2(x)
        
        
        return x



class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        
        
        output = self.fc_out(trg)
        
     
            
        return output, attention




class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
       
        
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
            
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        
        
       
        _trg = self.positionwise_feedforward(trg)
        
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        
        
        return trg, attention




class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
       
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)


        return src_mask
    
    def make_trg_mask(self, trg):
        
       
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
       
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
       
        
        return trg_mask

    def forward(self, src, trg):
        
  
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        
        
        enc_src = self.encoder(src, src_mask)
        
       
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
    
        
        return output, attention







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




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)




model1.apply(initialize_weights)




LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model1.parameters(), lr = LEARNING_RATE)




criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX) 




def train(model,iterator,optimizer,criterion,clip):
        
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
       
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
       
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)




def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:,:-1])
            
           
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
           
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs







model1 = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

model1.load_state_dict(torch.load(CWD+'/static/models/tut6-model.pt'))
model1.eval()

def translate_sentence(sentence, src_field, trg_field, model11, device, max_len = 50):
    
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model11.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model11.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model11.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model11.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention




example_idx = 8

src = vars(train_data.examples[example_idx])['src']
trg = vars(train_data.examples[example_idx])['trg']


translation, attention = translate_sentence(src, SRC, TRG, model1, device)

def listToString(s):  
    
    str1 = ""   
    for ele in s:  
        str1 += " "+ele     
    return str1  

german_result = translation[:-2]



class Translation_Predictor(nn.Module):
    def __init__(self, 
                 SRC, 
                 TRG, 
                 TRANS_MODEL, 
                 DEVICE_NAME):
        super().__init__()
        
        self.SRC_OBJ = SRC
        self.TRG_OBJ = TRG
        self.TRANS_MODEL_OBJ = TRANS_MODEL
        self.DEVICE_NAME_OBJ = DEVICE_NAME

TRANSLATION_OBJECT = Translation_Predictor(SRC,TRG,model1,device)

def get_translated_sentence(sentence):
    translation, attention = translate_sentence(sentence, TRANSLATION_OBJECT.SRC_OBJ, TRANSLATION_OBJECT.TRG_OBJ, TRANSLATION_OBJECT.TRANS_MODEL_OBJ, TRANSLATION_OBJECT.DEVICE_NAME_OBJ)
    german_result = translation[:-2]
    print('Translated sentence :',listToString(german_result))  
    return listToString(german_result)