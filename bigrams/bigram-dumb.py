import torch

namesFile = open("names.txt", "r")

names = namesFile.read().splitlines()

namesFile.close()

letterCombinations = {}

for name in names:
    name = "." + name + "."
    for index in range(0, len(name)-1):
        firstLetter = name[index]
        secondLetter = name[index+1]

        dictIndex = (firstLetter, secondLetter)

        letterCombinations[dictIndex] = letterCombinations.get(dictIndex, 0) + 1


dictKeys = sorted(letterCombinations.keys())

dict = {}

for i in ".abcdefghijklmnopqrstuvwxyz":
    dict[i] = []
    for a in ".abcdefghijklmnopqrstuvwxyz":
        dict[i].append({"letter": a, "probability": 0})


for key in dictKeys:

    index = None

    for i in range(len(dict[key[0]])):
        if dict[key[0]][i]['letter'] == key[1]:
            index = i

    dict[key[0]][index] = {"letter": key[1], "probability": letterCombinations[key]}

for i in dict:
    dict[i] = sorted(dict[i], key=lambda x: x["probability"])
    
    probabilitySum = sum([prob["probability"] for prob in dict[i]])

    for letterIndex in range(len(dict[i])):
        dict[i][letterIndex]['probability'] = dict[i][letterIndex]['probability'] / probabilitySum

# inference
currentIndex = 0

# 0 1 2 3 4 ... 27
# . a b c d ... z

dotToZProbabilities = []
dotToZLetters = []

for i in dict:
    tempDotToZ = sorted(dict[i], key=lambda x: x["letter"])
    tempDotToZLetters = [p['letter'] for p in tempDotToZ]
    tempDotToZProbabiblities = [p['probability'] for p in tempDotToZ]
    
    dotToZProbabilities.append(tempDotToZProbabiblities)
    dotToZLetters.append(tempDotToZLetters)

for i in range(50):
    out = []
    while True:
        possibilities = torch.tensor(dotToZProbabilities[currentIndex])

        currentIndex = torch.multinomial(possibilities, num_samples=1, replacement=True)[0].item()

        currentLetter = dotToZLetters[currentIndex][currentIndex]

        # print(currentIndex, currentLetter)

        out.append(currentLetter)

        if currentLetter  == ".":
            break

    print("".join(out))

