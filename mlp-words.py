import math
import torch
import matplotlib.pyplot as plt
import json

def vis():
    # visualize dimensions 0 and 1 of the embedding matrix C for all characters
    plt.figure(figsize=(8,8))
    plt.scatter(c[:,0].data, c[:,1].data, s=200)
    for i in range(c.shape[0]):
        plt.text(c[i,0].item(), c[i,1].item(), dictionary[i], ha="center", va="center", color='white')
    plt.grid('minor')
    plt.show()



# opening the names file and reading its data
file = open("output.txt", "r")

sentences = [x.replace(',', "").replace('.', "").replace('!', "").replace('?', "").replace('(', "").replace(')', "") for x in file.read().splitlines()]

file.close()

# creating a dataset of trigrams 
x = [] # input to the NN
y = [] # expected output

examplesCount = 0

gramSize = 10
embDim = 20

dict_json = open('dict.json', 'r')
dictionary = json.loads(dict_json.read())

x_json = open('xs.json', 'r')
x = json.loads(x_json.read())

y_json = open('ys.json', 'r')
y = json.loads(y_json.read())

g = torch.Generator().manual_seed(2147483647)

x = torch.tensor(x)
y = torch.tensor(y)

# creating all the params
c = torch.randn((len(dictionary),embDim), generator=g)

# creating hidden layer weights
wh = torch.randn((gramSize*embDim, 100), generator=g)
bh = torch.randn(100, generator=g)

# creating output layer
wo = torch.randn((100, len(dictionary)), generator=g)
bo = torch.randn(len(dictionary), generator=g)

params = [c, wh, bh, wo, bo]

for p in params:
    p.requires_grad = True

examplesCount = 32
loss = None

def sample(numofwords):
    # sampling from the model
    for _ in range(numofwords):
        sent = ("* "*gramSize).split()

        while sent[-1] != "*" or len(sent) == gramSize:
            context = [dictionary.index(x) for x in sent[-gramSize:]]
            emb = c[context]

            # calculating the hidden layer
            h = torch.tanh(emb.view(-1, gramSize*embDim) @ wh + bh)

            # calculating the output layer
            logits = h @ wo + bo
            counts = logits.exp()
            prob = counts / counts.sum(1, keepdim=True)

            nextWord = torch.multinomial(prob, 1)
            sent.append(dictionary[nextWord])

            # if len(sent) > 50:
            #     sent.append("*")


        print(" ".join(sent[gramSize:]))
        print("---------")

# training the model
for i in range(30000):
    minibatch = torch.randint(0, x.shape[0], (examplesCount,), generator=g)

    emb = c[x[minibatch]]

    # calculating the hidden layer
    h = torch.tanh(emb.view(-1, gramSize*embDim) @ wh + bh)

    # calculating the output layer
    logits = h @ wo + bo
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdim=True)

    # loss = -prob[torch.arange(examplesCount), y[minibatch]].log().mean()

    loss = torch.nn.functional.cross_entropy(logits, y[minibatch])

    if i % 1 == 0:
        print(f"Loss after {i} iterations is {loss.data}")
        sample(1)
    
    for p in params:
        p.grad = None
    
    loss.backward()

    delta = 0.15 if i < 10000 else 0.1
    for p in params:
        p.data -= delta * p.grad
        
print("Final loss = ", loss)


# vis()
sample(10)

