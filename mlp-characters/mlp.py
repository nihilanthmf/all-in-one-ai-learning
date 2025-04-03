import torch
import matplotlib.pyplot as plt

def vis():
    # visualize dimensions 0 and 1 of the embedding matrix C for all characters
    plt.figure(figsize=(8,8))
    plt.scatter(c[:,0].data, c[:,1].data, s=200)
    for i in range(c.shape[0]):
        plt.text(c[i,0].item(), c[i,1].item(), alph[i], ha="center", va="center", color='white')
    plt.grid('minor')
    plt.show()



# opening the names file and reading its data
namesFile = open("./mlp-characters/names.txt", "r")

names = namesFile.read().splitlines()

namesFile.close()

# creating a dataset of trigrams 
x = [] # input to the NN
y = [] # expected output

alph = list(".abcdefghijklmnopqrstuvwxyz")

examplesCount = 0

gramSize = 5
embDim = 10

for name in names:
    name = "." * gramSize + name + "."
    for index in range(gramSize, len(name)):
        context = [alph.index(x) for x in list(name[index-gramSize:index])]
        res = alph.index(name[index])

        x.append(context)
        y.append(res)

examplesCount = len(x)

g = torch.Generator().manual_seed(2147483647)

x = torch.tensor(x)
y = torch.tensor(y)

# creating all the params
c = torch.randn((27,embDim), generator=g)

# creating hidden layer weights
wh = torch.randn((gramSize*embDim, 100), generator=g)
bh = torch.randn(100, generator=g)

# creating output layer
wo = torch.randn((100, 27), generator=g)
bo = torch.randn(27, generator=g)

params = [c, wh, bh, wo, bo]

for p in params:
    p.requires_grad = True

examplesCount = 32

# training the model
for i in range(200000):
    minibatch = torch.randint(0, x.shape[0], (examplesCount,), generator=g)

    emb = c[x[minibatch]]

    # calculating the hidden layer
    h = torch.tanh(emb.view(-1, gramSize*embDim) @ wh + bh)

    # calculating the output layer
    logits = h @ wo + bo
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdim=True)

    loss = -prob[torch.arange(examplesCount), y[minibatch]].log().mean()
    
    if i % 1000 == 0:
        print(loss)

    if loss.data < 1.5:
        break
    
    for p in params:
        p.grad = None
    
    loss.backward()

    delta = 0.08 if i < 100000 else 0.008
    for p in params:
        p.data -= delta * p.grad
        
print(loss)

# sampling from the model
for _ in range(150):
    word = "."*gramSize

    while word[-1] != "." or len(word) == gramSize:
        context = [alph.index(x) for x in list(word[-gramSize:-1] + word[-1])]
        emb = c[context]   

        # calculating the hidden layer
        h = torch.tanh(emb.view(-1, gramSize*embDim) @ wh + bh)

        # calculating the output layer
        logits = h @ wo + bo
        counts = logits.exp()
        prob = counts / counts.sum(1, keepdim=True)

        nextLetter = torch.multinomial(prob, 1)
        word += alph[nextLetter]

    print(word[gramSize:])

# vis()


