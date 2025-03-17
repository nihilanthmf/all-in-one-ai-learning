import torch

# opening the names file and reading its data
namesFile = open("names.txt", "r")

names = namesFile.read().splitlines()

namesFile.close()

# alphabet to get the char's indices
alph = '.abcdefghijklmnopqrstuvwxyz'

# creating a letter combinations array (as well as the array with corresponding character indices)
letterCombinations = []

trainingData = []
evalData = []

for name in names[:1000]:
    name = "." + name + "."
    for index in range(0, len(name)-1):
        firstLetter = name[index]
        secondLetter = name[index+1]

        letterCombinations.append((firstLetter, secondLetter))
        trainingData.append(alph.index(firstLetter))
        evalData.append(alph.index(secondLetter))
    trainingData.append(alph.index(name[ - 1]))

trainingData = torch.tensor(trainingData)


# initialize the network
generator = torch.Generator().manual_seed(2147483647)
weights = torch.randn((27, 27), generator=generator, requires_grad=True)



# model training
for _ in range(50):
    # print(trainingData)
    inputEnc = torch.nn.functional.one_hot(trainingData, num_classes=27).float()
    # print(inputEnc)
    logits = inputEnc @ weights
    
    counts = logits.exp()

    # preditions = torch.tensor([[(item / row.sum()).item() for item in row] for row in counts], requires_grad=True)
    preditions = counts / counts.sum(dim=1, keepdim=True)

    # print(preditions)
    
    # calculate the loss function
    nllSum = 0
    nllCount = 0
    for rowIndex in range(len(evalData)):
        # calculcating the negative log likelihood for each predicted letter
        correctPrediction = evalData[rowIndex]
        modelLetterPred = preditions[rowIndex][correctPrediction]

        nll = -1 * torch.log(modelLetterPred)

        nllSum += nll
        nllCount += 1
    loss = nllSum / nllCount

    print(loss)

    # do the backpath for the loss function
    loss.backward()

    print('---')

    grad = weights.grad

    # actually tune the weights
    with torch.no_grad():
        weights -= 5 * weights.grad
    
# inference :)



