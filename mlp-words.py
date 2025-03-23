import torch
import matplotlib.pyplot as plt

def vis():
    # visualize dimensions 0 and 1 of the embedding matrix C for all characters
    plt.figure(figsize=(8,8))
    plt.scatter(c[:,0].data, c[:,1].data, s=200)
    for i in range(c.shape[0]):
        plt.text(c[i,0].item(), c[i,1].item(), dictionary[i], ha="center", va="center", color='white')
    plt.grid('minor')
    plt.show()



# opening the names file and reading its data
namesFile = open("sentences.txt", "r")

sentences = namesFile.read().splitlines()[:50000]

namesFile.close()

# creating a dataset of trigrams 
x = [] # input to the NN
y = [] # expected output

examplesCount = 0

gramSize = 3
embDim = 2

dictionary = ["<"]
for word in " ".join(sentences).split():
    dictionary.append(word)

print('dict created!')

dictionary = list(set(dictionary))

i=0
for sentence in sentences:
    sentence = "< " * gramSize + sentence + " <"
    sentence = sentence.split()

    i+=1
    print(i)

    for index in range(gramSize, len(sentence)):
        context = [dictionary.index(x) for x in list(sentence[index-gramSize:index])]
        res = dictionary.index(sentence[index])

        x.append(context)
        y.append(res)

examplesCount = len(x)

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

examplesCount = 320

print('started training')

# training the model
for i in range(10000):
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
    
    for p in params:
        p.grad = None
    
    loss.backward()

    delta = 0.1 if i < 10000 else 0.1
    for p in params:
        p.data -= delta * p.grad
        
print(loss)

# sampling from the model
for _ in range(5):
    sent = ("< "*gramSize).split()

    while sent[-1] != "<" or len(sent) == gramSize:
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

    print(" ".join(sent[gramSize:]))
    print("---------")
# vis()


