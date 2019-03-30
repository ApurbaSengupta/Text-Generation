import os
import unidecode
import string
import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_characters = string.printable
n_characters = len(all_characters)

# get data
all_files = ""

for file in os.listdir('./data'):
  all_files += unidecode.unidecode(open('./data/'+file).read()) + "\n"

file_len = len(all_files)

# use CUDA if available
use_cuda = False
if torch.cuda.is_available():
  use_cuda = True

chunk_len = 250

# get random chunk of data
def random_chunk(chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return all_files[start_index:end_index]

 # main model class
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
        self.dropout = nn.Dropout(0.3)
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

# turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    if use_cuda:
      tensor = tensor.cuda()
    return Variable(tensor)

# get random train data
def random_training_set(chunk_len=250):    
    chunk = random_chunk(chunk_len)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

# evaluate model
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

    # use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, states = decoder(prime_input[p], hidden, cell)
        
        if use_cuda:
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
          
    inp = prime_input[-1]
    loss = 0.
    
    for p in range(predict_len):
        output, states = decoder(inp, hidden, cell)
        
        if use_cuda:
          output = output.cuda()
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
        
        target = char_tensor(target_str[p])
        
        loss += criterion(output, target)
        
        # sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    loss_tot = total_loss(loss, predict_len)
    perplexity = perplexity_score(loss_tot)

    return predicted, loss_tot, perplexity

# define loss function
def total_loss(loss, predict_len):
    loss_tot = loss.cpu().item()/predict_len
    return loss_tot

# define perplexity
def perplexity_score(loss):
    perplexity = 2**loss
    return perplexity      

# helper function for time elapsed
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# train model
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

# generate text
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

    # use priming string to "build up" hidden state
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
        
        # sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # add predicted character to string and use as next input
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

# training
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
plt.show()

# evaluation
chunk = random_chunk(500)
prime_str, target_str = chunk[:251], chunk[251:]

gen_text, loss, perplexity = evaluate(target_str, prime_str, 250, temperature=0.8)

print("\nLoss: ", loss, " Perplexity:" , perplexity, "\n")
print("\n", gen_text, "\n")

# training evaluation

# Pride and Prejudice - Jane Austen
print(generate("\nThe tumult of her mind, was now painfully great. She knew not how \
to support herself, and from actual weakness sat down and cried for \
half-an-hour. ", 300, temperature=0.8))

# Dracula - Bram Stoker
print(generate("\nTo believe in things that you cannot. Let me illustrate. I heard once \
of an American who so defined faith: 'that faculty which enables us to \
believe things which we know to be untrue.' For one, I follow that man. ", 300, temperature=0.8))

# outside evaluation

# Emma - Jane Austen
print(generate("\nDuring his present short stay, Emma had barely seen him; but just enough \
to feel that the first meeting was over, and to give her the impression \
of his not being improved by the mixture of pique and pretension, now \
spread over his air.  ", 300, temperature=0.8))

# The Strange Case Of Dr. Jekyll And Mr. Hyde - Robert Louis Stevenson
print(generate("\nPoole swung the axe over his shoulder; the blow shook the building, and \
the red baize door leaped against the lock and hinges. A dismal \
screech, as of mere animal terror, rang from the cabinet. ", 300, temperature=0.8))
