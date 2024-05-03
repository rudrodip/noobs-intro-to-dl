import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
block_size = 3

def build_dataset(words):
  X, Y = [], []

  for w in words:
    context = [0] * block_size
    for ch in w:
      ix = stoi[ch]
      X.append(context)
      context = context[1:] + [ix]
      Y.append(ix)

  X = torch.tensor(X)
  Y = torch.tensor(Y)

  print(X.shape, Y.shape)
  return X, Y

random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xt, Yt = build_dataset(words[n2:])

# init params
n = 16
C = torch.randn((27, n))
W1 = torch.randn((n * block_size, 200))
b1 = torch.randn(200)
W2 = torch.randn((200, 27))
b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
  p.requires_grad = True


print(sum(p.nelement() for p in parameters))

# trackers
lossi = []
stepi = []

# training loop
epochs = 200000
batch_size = 100

for epoch in range(epochs):
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))

  # forward pass
  emb = C[Xtr[ix]]
  h = torch.tanh(emb.view(-1, n * block_size) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ytr[ix])
  print(f'loss = {loss.item()}')
  
  for p in parameters:
    p.grad = None

  loss.backward()

  lr = 0.1 if epoch < 100000 else 0.01
  for p in parameters:
    p.data -= lr * p.grad

  # track
  lossi.append(loss.log10().item())
  stepi.append(epoch)

plt.plot(stepi, lossi)

emb = C[Xtr]
h = torch.tanh(emb.view(-1, n * block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print(f'training loss={loss.item()}')

emb = C[Xdev]
h = torch.tanh(emb.view(-1, n * block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(f'dev loss={loss.item()}')

emb = C[Xt]
h = torch.tanh(emb.view(-1, n * block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yt)
print(f'test loss={loss.item()}')

# evaluation
for _ in range(20):
  out = []
  context = [0] * block_size

  while True:
    emb = C[torch.tensor([context])]
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
    context = context[1:] + [ix]
    out.append(ix)

    if ix == 0:
      break

  print(''.join(itos[i] for i in out))