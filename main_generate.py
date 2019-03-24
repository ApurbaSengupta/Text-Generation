import os
import unidecode
import string
import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
import time, math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_characters = string.printable
n_characters = len(all_characters)

all_files = ""

for file in os.listdir('./data'):
  all_files += unidecode.unidecode(open('./data/'+file).read()) + "\n"

file_len = len(all_files)

use_cuda = False
if torch.cuda.is_available():
  use_cuda = True

chunk_len = 250

def random_chunk(chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return all_files[start_index:end_index]

class TextGenerate(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bi=True):
        super(TextGenerate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bi = bi
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=self.bi)
        if self.bi:
          self.decoder = nn.Linear(hidden_size*2, output_size)
        else:
          self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(output_size, output_size)
    
    def forward(self, input, hidden, cell):
        input = self.encoder(input.view(1, -1))
        output, states = self.lstm(input.view(1, 1, -1), (hidden, cell))
        output = self.decoder(output.view(1, -1))
        output = self.dropout(output)
        output = self.out(output)
        return output, states

    def init_hidden(self):
        if self.bi:
          return Variable(torch.zeros(self.n_layers*2, 1, self.hidden_size))
        else:
          return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        
    def init_cell(self):
        if self.bi:
          return Variable(torch.zeros(self.n_layers*2, 1, self.hidden_size))
        else:
          return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    if use_cuda:
      tensor = tensor.cuda()
    return Variable(tensor)

def random_training_set(chunk_len=250):    
    chunk = random_chunk(chunk_len)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

def evaluate(target_str, prime_str='A', predict_len=100, temperature=0.8):
    decoder.load_state_dict(torch.load('./model_generate.pt'))
    decoder.eval()
    
    hidden = decoder.init_hidden()
    cell = decoder.init_cell()
    
    if use_cuda:
      hidden = hidden.cuda()
      cell = cell.cuda()
    
    prime_input = char_tensor(prime_str)
    predicted = prime_str + "\n-------->\n"

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, states = decoder(prime_input[p], hidden, cell)
        
        if use_cuda:
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
          
    inp = prime_input[-1]
    loss = 0
    predicted_next = ''
    
    for p in range(predict_len):
        output, states = decoder(inp, hidden, cell)
        
        if use_cuda:
          output = output.cuda()
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
        
        target = char_tensor(target_str[p])
        
        loss += criterion(output, target)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)
        predicted_next += predicted_char
    
    targ = char_tensor(target_str).cpu().numpy()
    pred = char_tensor(predicted_next).cpu().numpy()

    loss_tot = total_loss(loss, predict_len)
    f1 = F1_score(targ, pred)
    bleu = BLEU_score([list(target_str)], list(predicted_next))

    return predicted, loss_tot, f1, bleu

def total_loss(loss, predict_len):
    loss_tot = loss.cpu().item()/predict_len
    return loss_tot

def F1_score(target, predicted):
    f1 = f1_score(target, predicted, average='micro')
    return f1

def BLEU_score(target, predicted):
    bleu = sentence_bleu(target, predicted, weights=(0, 0.333, 0.333, 0.333))
    return bleu        

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(inp, target):
    decoder.train()
    target.unsqueeze_(-1)
    hidden = decoder.init_hidden()
    cell = decoder.init_cell()
    
    if use_cuda:
      hidden = hidden.cuda()
      cell = cell.cuda()
    
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, states = decoder(inp[c], hidden, cell)
        if use_cuda:
          output = output.cuda()
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
        loss += criterion(output, target[c])

    loss.backward()
    decoder_optimizer.step()
    
    torch.save(decoder.state_dict(), './model_generate.pt')
    
    loss_tot = total_loss(loss, chunk_len)

    return loss_tot

def generate(prime_str='A', predict_len=100, temperature=0.8):
    decoder.load_state_dict(torch.load('./model_generate.pt'))
    decoder.eval()
    
    hidden = decoder.init_hidden()
    cell = decoder.init_cell()
    
    if use_cuda:
      hidden = hidden.cuda()
      cell = cell.cuda()
    
    prime_input = char_tensor(prime_str)
    predicted = prime_str + "\n--------->\n"

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, states = decoder(prime_input[p], hidden, cell)
        
        if use_cuda:
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
          
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, states = decoder(inp, hidden, cell)
        
        if use_cuda:
          output = output.cuda()
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)
    
    return predicted

n_epochs = 25000
print_every = 2500
plot_every = 250
hidden_size = 100
n_layers = 2
lr = 0.0005
bi = True

decoder = TextGenerate(n_characters, hidden_size, n_characters, n_layers, bi)
if use_cuda:
  decoder = decoder.cuda()
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
  
    loss = train(*random_training_set(chunk_len))
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

plt.figure()
plt.plot(all_losses)

chunk = random_chunk(500)
prime_str, target_str = chunk[:251], chunk[251:]

gen_text, loss, f1, bleu = evaluate(target_str, prime_str, 250, temperature=0.8)

print("\nLoss: ", loss, " F1: ", f1, " BLEU: ", bleu, "\n")
print("\n", gen_text, "\n")

print(generate("\nI was very distraught. I called my friend ", 250, temperature=0.8))

print(generate("\nI never wanted this to happen. But the ", 250, temperature=0.8))
