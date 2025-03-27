import csv
import torch
import random

# reading and encoding the data
passengers = []
survived = []
passengers_test = []
survived_test = []

with open('titanic/train.csv', newline='') as csvfile:
    spamreader = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
    # random.shuffle(spamreader) loss equals to ~0.4695

    training_dataset = spamreader[:int(1*len(spamreader))]
    test_dataset = spamreader[int(1*len(spamreader)):]

    for row in spamreader:

        socioecclass = int(row[2])
        sex = int(1 if row[4] == 'male' else 0)
        age =  int(float(row[5])) if row[5] != "" else 100
        sibSp = int(row[6])
        parch = int(row[7])

        fare = float(row[9])
        if fare > 50:
            fare = 0
        elif fare > 25:
            fare = 1
        else:
            fare = 2

        cabin = row[10]
        cabin = cabin.replace("A", '1.').replace("B", '2.').replace("C", '3.').replace("D", '4.').replace("E", '5.').replace("F", '6.').replace("G", '7.').replace("T", '8.').split()[0] if cabin != "" else 0

        cabin = float(cabin) 

        embarked = row[11]
        if embarked == "C":
            embarked = 0
        elif embarked == "Q":
            embarked = 1
        else:
            embarked = 2
        
        if row in training_dataset:
            passengers.append([socioecclass, sex, age, sibSp, parch, fare, embarked, cabin])
            survived.append(int(row[1]))
        else:
            passengers_test.append([socioecclass, sex, age, sibSp, parch, fare, embarked, cabin])
            survived_test.append(int(row[1]))

passengers = torch.tensor(passengers).float()
survived = torch.tensor(survived)

passengers_test = torch.tensor(passengers_test).float()
survived_test = torch.tensor(survived_test)

# initialising the model
g = torch.Generator().manual_seed(100)
num_of_neurons = 200

wh = torch.randn((8, num_of_neurons), generator=g)
bh = torch.randn((num_of_neurons), generator=g)

wo = torch.randn((num_of_neurons, 2), generator=g)
bo = torch.randn((2), generator=g)

params = [wh, bh, wo, bo]

for p in params:
    p.requires_grad = True

# training the network
for i in range(500000):
    # forward pass
    h = torch.tanh(passengers @ wh + bh)

    logits = h @ wo + bo

    loss = torch.nn.functional.cross_entropy(logits, survived)

    if i % 1000 == 0:
        print(loss)

    # backward pass + params updates
    for p in params:
        p.grad = None

    loss.backward()

    learning_rate = 0.012
    if loss.data < 0.3:
        learning_rate = 0.009
    elif loss.data < 0.2:
        learning_rate = 0.009

    for p in params:
        p.data -= learning_rate * p.grad

# calculating the loss on the test dataset
# h = torch.tanh(passengers_test @ wh + bh)

# logits = h @ wo + bo

# loss = torch.nn.functional.cross_entropy(logits, survived_test)

# print("test dataset loss: ", loss)

# uploading the results to the csv file
guesses = [
    ["PassengerId", "Survived"]
]
with open('titanic/test.csv', newline='') as csvfile:
    spamreader = list(csv.reader(csvfile, delimiter=',', quotechar='|'))

    for row in spamreader:
        socioecclass = int(row[1])
        sex = int(1 if row[3] == 'male' else 0)
        age =  int(float(row[4])) if row[4] != "" else 100
        sibSp = int(row[5])
        parch = int(row[6])
        passId = int(row[0])

        fare = float(row[8]) if row[8] != "" else 2
        if fare > 50:
            fare = 0
        elif fare > 25:
            fare = 1
        else:
            fare = 2

        cabin = row[9]
        cabin = cabin.replace("A", '1.').replace("B", '2.').replace("C", '3.').replace("D", '4.').replace("E", '5.').replace("F", '6.').replace("G", '7.').replace("T", '8.').split()[0] if cabin != "" else 0

        cabin = float(cabin) 

        embarked = row[10]
        if embarked == "C":
            embarked = 0
        elif embarked == "Q":
            embarked = 1
        else:
            embarked = 2
        
        passenger_data = torch.tensor([socioecclass, sex, age, sibSp, parch, fare, embarked, cabin]).float()

        # getting the predictions 
        h = torch.tanh(passenger_data @ wh + bh)

        logits = h @ wo + bo

        counts = list(logits.exp())

        guesses.append([passId, counts.index(max(counts))])

with open("output.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(guesses)
