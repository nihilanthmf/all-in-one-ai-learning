# import json

# with open('train.json', 'r') as json_file:
#     data = json.loads(json_file.read())

# output = []

# for d in data["sentences"]:
#     output.append(d["sentence"])
#     output.append(d["paragraph"])

# with open('output.txt', 'w') as txt_file:
#     for value in output:
#         txt_file.write(f"{value}\n")

import json
# opening the names file and reading its data
namesFile = open("output.txt", "r")

sentences = [x.replace(',', "").replace('.', "").replace('!', "").replace('?', "").replace('(', "").replace(')', "") for x in namesFile.read().splitlines()][:5000]

namesFile.close()

# creating a dataset of trigrams 
x = [] # input to the NN
y = [] # expected output

examplesCount = 0

gramSize = 10
embDim = 2

dictionary = ["*", ".", ",", "!", '?']
for word in " ".join(sentences).split():
    dictionary.append(word)

dictionary = list(set(dictionary))

with open('dict.json', 'w') as f:
    json.dump(dictionary, f, indent=4)

i=0
for sentence in sentences:
    sentence = "* " * gramSize + sentence + " *"
    sentence = sentence.split()

    i+=1
    print(i)

    for index in range(gramSize, len(sentence)):
        context = [dictionary.index(x) for x in list(sentence[index-gramSize:index])]
        res = dictionary.index(sentence[index])

        x.append(context)
        y.append(res)

with open('xs.json', 'w') as f:
    json.dump(x, f, indent=4)

with open('ys.json', 'w') as f:
    json.dump(y, f, indent=4)