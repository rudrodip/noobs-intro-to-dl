import torch
import torch.nn.functional as F

words = open("names.txt").read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# training set
xs, ys = [], []

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]

    xs.append(ix1)
    ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

num = xs.nelement()

# NN init
W = torch.randn((27, 27), requires_grad=True)
LEARNING_RATE = 50

for k in range(200):
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float()
  logits = xenc @ W
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdim=True)
  loss = -probs[torch.arange(num), ys].log().mean()
  print(loss.item())

  # backward pass
  W.grad = None
  loss.backward()

  # tune
  W.data += -LEARNING_RATE * W.grad


# sampling
for i in range(5):
  word = ''
  ix = 0

  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)

    ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
    word += itos[ix]
    if ix == 0:
      break
  print(word)