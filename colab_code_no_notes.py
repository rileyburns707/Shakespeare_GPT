import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

print("length of dataset in characters: ", len(text))
print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars)) 
print(vocab_size)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(encode("hii there"))
print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

block_size = 8 
train_data[:block_size+1]

x = train_data[:block_size] 
y = train_data[1:block_size+1] 
for t in range(block_size): 
  context = x[:t+1] 
  target = y[t]
  print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)
batch_size = 4 
block_size = 8

def get_batch(split):
 
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) 
  x = torch.stack([data[i:i+block_size] for i in ix]) 
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
  return x, y

xb, yb = get_batch('train') 
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('----')

for b in range(batch_size): 
  for t in range(block_size): 
    context = xb[b, :t+1]
    target = yb[b,t]
    print(f"When input is {context.tolist()} the target: {target}")


print(xb)


# Bigram Language Model
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets = None):
    logits = self.token_embedding_table(idx)
     if targets is None:
       loss = None
     else:
        B,T,C = logits.shape 
        logits = logits.view(B*T, C) 
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
  return logits, loss

  def generate(self, idx, max_new_tokens):
      for _ in range(max_new_tokens):
          logits, loss = self(idx) 
          logits = logits[:, -1, :] 
          probs = F.softmax(logits, dim = -1) 
          idx_next = torch.multinomial(probs, num_samples = 1) 
          idx = torch.cat((idx, idx_next), dim = 1)
      return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb) 
print(logits.shape)

print(loss) 

idx = torch.zeros((1,1), dtype=torch.long)

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(100): 
  xb, yb, = get_batch('train') 
  logits, loss = m(xb,yb)
  optimizer.zero_grad(set_to_none=True) 
  loss.backward()
  optimizer.step() 
  
print(loss.item())


batch_size = 32
for steps in range(1000):
  xb, yb, = get_batch('train')
  logits, loss = m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  
print(loss.item())


batch_size = 32
for steps in range(10000):
  xb, yb, = get_batch('train')
  logits, loss = m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  
print(loss.item())

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=400)[0].tolist()))

torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
X = torch.randn(B,T,C)

tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ X

out.shape

torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
k = key(x) 
q = query(x) 

wei = q @ k.transpose(-2,-1) 
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ X

out.shape

wei

wei[0]


torch.manual_seed(1337)
B,T,C = 4,8,32 
x = torch.randn(B,T,C)

# single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) 
q = query(x) 

wei = q @ k.transpose(-2,-1) 
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x) 
out = wei @ v 

out.shape



# scaled attention
k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2,-1)  * head_size**-0.5

k.var()
q.var()
wei.var()

torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)
torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])*8, dim=-1)


class BatchNorm1d:
  def __init__(self, dim, eps=1e-5, momentim=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

    def parameters(self):
      return [self.gamma, self.beta]

torch.manual_seed(1337)
module = BatchNorm1d(100)
x = torch.randn(32, 100) # batch size 32 of 100 dimensional vectors
x = module(x)
x.shape

x[:,0].mean(), x[:,0].std()

x[0,:].mean(), x[0,:].std()





















