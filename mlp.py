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
namesFile = open("names.txt", "r")

names = namesFile.read().splitlines()

namesFile.close()

# creating a dataset of trigrams 
x = [] # input to the NN
y = [] # expected output

alph = list(".abcdefghijklmnopqrstuvwxyz")

examplesCount = 0

for name in names:
    name = "..." + name + "."
    for index in range(3, len(name)):
        context = [alph.index(x) for x in list(name[index-3:index])]
        res = alph.index(name[index])

        x.append(context)
        y.append(res)

examplesCount = len(x)

x = torch.tensor(x)
y = torch.tensor(y)

# creating all the params
c = torch.randn((27,2))

# creating hidden layer weights
wh = torch.randn((6, 100))
bh = torch.randn(100)

# creating output layer
wo = torch.randn((100, 27))
bo = torch.randn(27)

params = [c, wh, bh, wo, bo]

for p in params:
    p.requires_grad = True

examplesCount = 320

# training the model
for i in range(20000):
    minibatch = torch.randint(0, x.shape[0], (examplesCount,))

    emb = c[x[minibatch]]

    # calculating the hidden layer
    h = torch.tanh(emb.view(-1, 6) @ wh + bh)

    # calculating the output layer
    logits = h @ wo + bo
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdim=True)

    loss = -prob[torch.arange(examplesCount), y[minibatch]].log().mean()
    
    for p in params:
        p.grad = None
    
    loss.backward()

    delta = 0.3 if i < 100 else 0.1
    for p in params:
        p.data -= delta * p.grad
        
print(loss)

# sampling from the model
for _ in range(5):
    word = "..."

    while word[-1] != "." or len(word) == 3:
        context = [alph.index(x) for x in list(word[-3:-1] + word[-1])]
        emb = c[context]   

        # calculating the hidden layer
        h = torch.tanh(emb.view(-1, 6) @ wh + bh)

        # calculating the output layer
        logits = h @ wo + bo
        counts = logits.exp()
        prob = counts / counts.sum(1, keepdim=True)

        nextLetter = torch.multinomial(prob, 1)
        word += alph[nextLetter]

    print(word[3:])

vis()


