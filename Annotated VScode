import torch
import torch.nn as nn
from torch.nn import functional as F

# # hyperparameters
# batch_size = 64 # how many independent sequences will we process in parallel?
#     # was 32 but changed it when we scaled up model
# block_size = 256 # what is the maxium context length for predictions?
#     # was 8 but changed it when we scaled up model. So we have 256 context characters to predict the 257th term instead of just 8
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
#     # was 1e-3 but changed it when we scaled up model. Lowered it because neural net is much bigger
# device = 'cuda' if torch.cuda.is_available() else 'cpu' # adds ability to run on gpu if you have it, this makes it a lot faster
# eval_iters = 200
# n_embd = 384 # n_embd = number of embedding dimensions
#     # was 32 but changed it when we scaled up model
# n_head = 6 # every head is 64 dimensional
# n_layer = 6
# dropout = 0.2 # every foward/backward pass 20% of these intermediate calculations are disabled and dropped to 0
# # -----------------
# # I have a macbook so to run this without a cpu would not be a good idea lol. But if you had a GPU these numbers 
# # would take about 15-30 minutes to run and you would get a very solid shakespeare play! But for me I have to turn 
# # these numbers down unfortunently so my output won't be nearly as good, but it is great for learning!

# # these worked but too ~10 minutes. I commented the output below thr dropout
# batch_size = 32 # Moderate batch size
# block_size = 128 # Increased context length but still manageable
# max_iters = 3000 # Reasonable number of iterations
# eval_interval = 300 # Evaluate periodically
# learning_rate = 5e-4 # Moderate learning rate for stability
# device = 'cpu' # Force usage of CPU
# eval_iters = 100 # Moderate number of evaluation iterations
# n_embd = 128 # Moderate number of embedding dimensions
# n_head = 4 # Moderate number of attention heads
# n_layer = 4 # Moderate number of layers
# dropout = 0.1 # Moderate dropout rate

# BOLIXO:
# What have be is yet speech is fold, geet here wento
# The righter onseder'd. Apola nuse;
# For cholow for this my fouriol plong,
# And, so I sir'd your lidsby't, and my yie,
# I was I repter endyd have him counleeds
# Mersomishink, ell be this defere? What thou cheenced.

# TRYON:

# Gaver:
# Moth sent me, stayer delled pringss: well enough here are,
# Thy seet
# most in's bouring, frief my not,
# Laar of and to prud I deatinung him was a stor murch
# Datablemish greal hander: he descopt, he may.

# the form is right but the results are nonsensical, but shows at more scale what is possible


# hyperparameters that can work for me
batch_size = 24 # Slightly lower batch size
block_size = 96 # Lower context length but still usable
max_iters = 2000 # Reduced number of iterations
eval_interval = 200 # Evaluate periodically
learning_rate = 1e-3 # Maintain moderate learning rate for faster training
device = 'cpu' # Force usage of CPU
eval_iters = 100 # Moderate number of evaluation iterations
n_embd = 96 # Reduced number of embedding dimensions
n_head = 3 # Reduced number of attention heads
n_layer = 3 # Reduced number of layers
dropout = 0.1 # Maintain moderate dropout rate




torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()


# all the unique characters that occur in this text
chars = sorted(list(set(text))) 
vocab_size = len(chars) 

# create a mapping from characters that ocur in this text
stoi = { ch:i for i, ch in enumerate(chars) } 
itos = { i:ch for i, ch in enumerate(chars) } 
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# train test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% of data is train, rest is test data
train_data = data[:n]
val_data = data[n:] 


# data loading
def get_batch(split):

  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) 
  x = torch.stack([data[i:i+block_size] for i in ix]) 
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
  x, y = x.to(device), y.to(device)
  return x, y


# this function averages out the loss over multiple batches
@torch.no_grad() # tells PyTorch everything in this function will not call .backward on. Makes it more efficent.
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



## Head Module: implements a single head of self-attention
class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) # (B, T, C)
    q = self.query(x) # (B, T, C)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wei =  F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B, T, C)
    out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
    return out



# multi-head attention: multiple heads of self attention in parallel
class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size): # however many heads you want then what is the size of each
    super().__init__()
    self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
    # run all the heads in parallel into a list
 
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim =-1)
    out = self.dropout(self.proj(out)) # projection is the linear transformation of the out come of layer torch.cat([h(x) for h in self.heads], dim =-1)
    # projection back into the residual pathway
    return out
    # concatenate all of the outputs over the channel dimension


# want to add compuation on a per node level into the network
class FeedForward(nn.Module):
  # a linear layer followed by a non-linearity

  def __init__(self, n_embd):
    super().__init__() # Calls the constructor of the parent class (nn.Module) to initialize the base class. This is necessary to properly set up PyTorch modules
    self.net = nn.Sequential(     # Creates a sequential container (nn.Sequential), which allows you to define a neural network with a sequence of layers. self.net will hold this sequence.
      nn.Linear(n_embd, 4 * n_embd),  # adds a layer to the sequential container
      nn.ReLU(), # ReLU = Rectified Linear Unit. Adds ReLU to sequential container. ReLU sets all negative values to zero and leaves positive values unchanged.
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout), # dropout can be added right before the connection back into the residual pathway 
    )
    

  def forward(self,x):
    return self.net(x) # pass the input through the sequential container

# feed forward allows each token to 'think' on the data it just collected from self-attention


# Transformer Block: communication followed by computation. Combines the previous few classes
class Block(nn.Module):

  def __init__(self, n_embd, n_head): # n_embd = embedding dimension. n_head = number of heads we'd like
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size) # communication
    self.ffwd = FeedForward(n_embd) # computation
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd) # 2 layer norms and we tell it the embedding dimension
      # this layering is per token transformation (happens to all 32 tokens) that
      # normalizes the features and makes them unit mean and unit Gaussian at initialization
      # after the model runs it might create layer norms that are not unti Gaussian, but the
      # optimization will determine that

  
  def forward(self,x):
    x = x + self.sa(self.ln1(x)) # fork off main path to do computation, then come back to x
    x = x + self.ffwd(self.ln2(x)) # computation... fork off like line above
    # we are applying layer norm before the transformation, called prenorm formulation
    return x


# simple bigram model        
class BigramLanguageModel(nn.Module): 

  def __init__(self):
    super().__init__()
   # each token directly reads off the logits for the next token form a loopup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding table with 32 dimensional embeddings
    self.position_embedding_table = nn.Embedding(block_size, n_embd) # each position will get its own embedding vector
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)


  def forward(self, idx, targets = None): 
    B, T = idx.shape # decodes B by T from idx.shape
    # idx and targets are both (B, T) tensor of integers

    tok_emb = self.token_embedding_table(idx)  # (B, T, C = n_embd). C = n_embd.  tok_emb = token embeddings
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # integers from 0 to T-1. (T, C)
    x = tok_emb + pos_emb # (B, T, C). x holds the token identities and the postition in which they occur
    x = self.blocks(x) # (B, T, C)
    x = self.ln_f(x) # (B, T, C)
    logits = self.lm_head(x) # (B, T, C = vocab_size). Not the same C as line above

    if targets is None: 
      loss = None
    else:
        B,T,C = logits.shape 
        logits = logits.view(B*T, C) 
        targets = targets.view(B*T) 
        loss = F.cross_entropy(logits, targets) 

    return logits, loss


  def generate(self, idx, max_new_tokens):
      # idx is (B, T) array of indices in the current context
      for _ in range(max_new_tokens):
          # cop idx to the last block_size tokens
          idx_cond = idx[:,-block_size:]
          # get the predictions
          logits, loss = self(idx_cond) # loss is ignored here
          # focus only on the last time step
          logits = logits[:, -1, :] # becomes (B, C)
          # apply softmax to get probabilties
          probs = F.softmax(logits, dim = -1) # (B, C)
          # sample from the distribution
          idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
          # append sampled index to the running sequence
          idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
      return idx


model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 

for iter in range(max_iters):
   
   # every once in a while evaluate the loss on train and val sets
   if iter % eval_interval == 0:
       losses = estimate_loss()
       print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
   
   # sample a batch of data
   xb, yb = get_batch('train') 
   

   # evaluate the loss
   logits, loss = model(xb,yb) 
   optimizer.zero_grad(set_to_none=True) 
   loss.backward() 
   optimizer.step() 

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
