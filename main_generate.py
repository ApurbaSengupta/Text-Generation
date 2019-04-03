import os
import unidecode
import string
import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
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

# define length of string to consider while training
chunk_len = 250

# get a random chunk of data of length 'chunk_len'
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
        self.out = nn.Linear(output_size, output_size)  
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input, hidden, cell):

        # encoder
        input = self.encoder(input.view(1, -1))
        input = self.dropout(input)
        output, states = self.lstm(input.view(1, 1, -1), (hidden, cell))
        output = output.permute(1, 0, 2)

        # attention
        if self.bi:
          out1, out2 = output[:,:,:self.hidden_size], output[:,:,self.hidden_size:]
          h1, h2 = states[0][states[0].size()[0] - 2,:,:], states[0][states[0].size()[0] - 1,:,:]
          attn_wts_1 = F.softmax(torch.bmm(out1, h1.unsqueeze(2)).squeeze(2), 1)
          attn_wts_2 = F.softmax(torch.bmm(out2, h2.unsqueeze(2)).squeeze(2), 1)
          attn_1 = torch.bmm(out1.transpose(1, 2), attn_wts_1.unsqueeze(2)).squeeze(2)
          attn_2 = torch.bmm(out2.transpose(1, 2), attn_wts_2.unsqueeze(2)).squeeze(2)
          attn = torch.cat((attn_1, attn_2), 1)

        else:
          h = states.squeeze(0)
          attn_wts = F.softmax(torch.bmm(output, h.unsqueeze(2)).squeeze(2), 1)
          attn = torch.bmm(output.transpose(1, 2), attn_wts.unsqueeze(2)).squeeze(2)
        
        # decoder
        output = self.decoder(attn)
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

# get random training data
def random_training_set(chunk_len=250):    
    chunk = random_chunk(chunk_len)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

# evaluate model
def evaluate(target_str, prime_str='A', predict_len=100, temperature=0.8):
    model.load_state_dict(torch.load('./model_generate.pt'))
    model.eval()
    
    hidden = model.init_hidden()
    cell = model.init_cell()
    
    if use_cuda:
      hidden = hidden.cuda()
      cell = cell.cuda()
    
    prime_input = char_tensor(prime_str)
    predicted = prime_str + "\n-------->\n"

    # use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, states = model(prime_input[p], hidden, cell)
        
        if use_cuda:
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
          
    inp = prime_input[-1]
    loss = 0.
    
    for p in range(predict_len):
        output, states = model(inp, hidden, cell)
        
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

# get loss
def total_loss(loss, predict_len):
    loss_tot = loss.cpu().item()/predict_len
    return loss_tot

# get perplexity
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
    model.train()
    target.unsqueeze_(-1)
    hidden = model.init_hidden()
    cell = model.init_cell()
    
    if use_cuda:
      hidden = hidden.cuda()
      cell = cell.cuda()
    
    model.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, states = model(inp[c], hidden, cell)
        if use_cuda:
          output = output.cuda()
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
        loss += criterion(output, target[c])

    loss.backward()
    model_optimizer.step()
    
    torch.save(model.state_dict(), './model_generate.pt')
    
    loss_tot = total_loss(loss, chunk_len)
    perplexity = perplexity_score(loss_tot)

    return loss_tot, perplexity

# generate text given context
def generate(prime_str='A', predict_len=100, temperature=0.8):
    model.load_state_dict(torch.load('./model_generate.pt'))
    model.eval()
    
    hidden = model.init_hidden()
    cell = model.init_cell()
    
    if use_cuda:
      hidden = hidden.cuda()
      cell = cell.cuda()
    
    prime_input = char_tensor(prime_str)
    predicted = prime_str + "\n--------->\n"

    # use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, states = model(prime_input[p], hidden, cell)
        
        if use_cuda:
          hidden, cell = states[0].cuda(), states[1].cuda()
        else:
          hidden, cell = states[0], states[1]
          
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, states = model(inp, hidden, cell)
        
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

# main
if __name__ == "__main__":

    n_epochs = 25000
    print_every = 2500
    plot_every = 100
    hidden_size = 100
    n_layers = 2
    lr = 0.0005
    bi = True

    # define model
    model = TextGenerate(n_characters, hidden_size, n_characters, n_layers, bi)
    if use_cuda:
      model = model.cuda()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # # train the model
    # start = time.time()
    # all_losses = []
    # all_perplexities = []
    # loss_avg = 0.
    # perplexity_avg = 0.

    # for epoch in range(1, n_epochs + 1):
      
    #     loss, perplexity = train(*random_training_set(chunk_len))
    #     loss_avg += loss
    #     perplexity_avg += perplexity

    #     if epoch % print_every == 0:
    #         print('[%s taken (%d epochs %d%% trained) Loss: %.4f Perplexity: %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss, perplexity))

    #     if epoch % plot_every == 0:
    #         all_losses.append(loss_avg / plot_every)
    #         all_perplexities.append(perplexity_avg / plot_every)
    #         loss_avg = 0.
    #         perplexity_avg = 0.

    # plt.figure()
    # plt.plot(all_losses)
    # plt.show()

    # plt.figure()
    # plt.plot(all_perplexities)
    # plt.show()

    # evaluation
    l = 0.
    p = 0.
    for i in range(1000):
      chunk = random_chunk(500)
      prime_str, target_str = chunk[:251], chunk[251:]

      gen_text, loss, perplexity = evaluate(target_str, prime_str, 250, temperature=0.8)
      l += loss
      p += perplexity
    print("\nLoss: ", l/1000, " Perplexity:" , p/1000, "\n")
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
