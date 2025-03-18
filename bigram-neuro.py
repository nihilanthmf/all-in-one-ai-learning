import torch

# opening the names file and reading its data
namesFile = open("words_alpha.txt", "r")

names = namesFile.read().splitlines()

namesFile.close()

# alphabet to get the char's indices
alph = ".abcdefghijklmnopqrstuvwxyz"
alphLen = len(alph)

trainingData = []
evalData = []

for name in names:
    name = "." + name + "."
    for index in range(0, len(name)-1):
        firstLetter = name[index]
        trainingData.append(alph.index(firstLetter))

        secondLetter = name[index+1]
        evalData.append(alph.index(secondLetter))
    # trainingData.append(alph.index(name[-1]))

trainingData = torch.tensor(trainingData)
evalData = torch.tensor(evalData)

# initialize the network
generator = torch.Generator().manual_seed(2147483647)
weights = torch.randn((alphLen, alphLen), generator=generator, requires_grad=True)



# model training
for i in range(1, 100):
    inputEnc = torch.nn.functional.one_hot(trainingData, num_classes=alphLen).float()

    logits = inputEnc @ weights
    
    counts = logits.exp()

    # preditions = torch.tensor([[(item / row.sum()).item() for item in row] for row in counts], requires_grad=True)
    preditions = counts / counts.sum(dim=1, keepdim=True)
    
    # calculate the loss function
    loss = -preditions[torch.arange(trainingData.nelement()), evalData].log().mean()

    print(loss)

    # do the backpath for the loss function
    loss.backward()

    grad = weights.grad

    # actually tune the weights
    weights.data -= 0.1 * weights.grad
    
# inference :)
for _ in range(500):
    out = ""
    letter = ""
    maxPredictionIndex = 0

    while letter != ".":
        inputEnc = torch.nn.functional.one_hot(torch.tensor([maxPredictionIndex]), num_classes=alphLen).float()

        logits = inputEnc @ weights
            
        counts = logits.exp()

        # preditions = torch.tensor([[(item / row.sum()).item() for item in row] for row in counts], requires_grad=True)
        preditions = counts / counts.sum(dim=1, keepdim=True)

        maxPredictionIndex = torch.multinomial(preditions, 1, replacement=False, generator=None, out=None)
        #preditions.tolist()[0].index(preditions.max())
        letter = alph[maxPredictionIndex]
        out += letter
    print(out)

