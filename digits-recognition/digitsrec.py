import torch
import csv
import random

# loss
# 0.4032
# 1.1678 - 500 neurons
# 0.3626 - 100 neurons
# 0.4456 - 150 neurons
# 0.3513 - 50 neurons
# 0.3767 - 25 neurons
# 0.3835 - 75 neurons

# reading the data
f = open("./digits-recognition/digits.txt")
content = f.readlines()
random.shuffle(content)
f.close()

f = open("./digits-recognition/digits-test.txt")
content_test = f.readlines()
f.close()

images = []
ans = []
images_test = []
ans_test = []

full_dataset = []
# creating full dataset
for img in content_test[:len(content_test)]:
    imgspl = img.split(",")
    full_dataset.append([1 if int(x)> 10 else 0 for x in imgspl])

# creating train dataset
for img in content[:int(0.95*len(content))]:
    imgspl = img.split(",")
    ans.append(int(imgspl[0]))
    images.append([1 if int(x)> 10 else 0 for x in imgspl[1:]])

# creating test dataset
for img in content[int(0.95*len(content)):]:
    imgspl = img.split(",")
    ans_test.append(int(imgspl[0]))
    images_test.append([1 if int(x)> 10 else 0 for x in imgspl[1:]])

images = torch.tensor(images).float()
ans = torch.tensor(ans)

images_test = torch.tensor(images_test).float()
ans_test = torch.tensor(ans_test)

full_dataset = torch.tensor(full_dataset).float()



# initializing the network's weights and biases
g = torch.Generator().manual_seed(1235355345)

wh = torch.randn((784, 50), generator=g) / 784 ** 0.5
bh = torch.randn((50), generator=g) * 0

wo = torch.randn((50, 10), generator=g) * 0.01
bo = torch.randn((10), generator=g) * 0

params = [wh, bh, wo, bo]

# requiring gradients on each param
for p in params:
    p.requires_grad = True


# initializing the model function
def model(input):
    h = torch.sigmoid(input @ wh + bh)
    logits = h @ wo + bo

    return logits

# training the network
miniBatchCount = 200 # make it 2000 later on
for i in range(300000):
    minibatch = torch.randint(0, images.shape[0], (miniBatchCount,), generator=g)

    logits = model(images[minibatch])

    loss = torch.nn.functional.cross_entropy(logits, ans[minibatch])

    if i % 1000 == 0:
        print(loss.data)

    for p in params:
        p.grad = None

    loss.backward()

    learningAlpha = 0.2

    if loss.data < 0.75:
        learningAlpha = 0.1
    if loss.data < 0.5:
        learningAlpha = 0.05
    if loss.data < 0.2:
        learningAlpha = 0.01
    if loss.data < 0.1:
        learningAlpha = 0.007

    for p in params:
        p.data -= learningAlpha * p.grad


def calc_train_dataset_results():
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, ans)

    # Get predicted class (assuming y_pred contains raw logits or probabilities)
    y_pred_labels = torch.argmax(logits, dim=1)
        
    # Compare predictions with true labels
    correct = (y_pred_labels == ans).sum().item()
        
    # Compute accuracy
    accuracy = correct / ans.size(0)

    print("entire dataset loss: ", loss.data)
    print("entire dataset accuracy: ", accuracy)
calc_train_dataset_results()


def sample():
    print("Guesses:")
    for i in range(int(0.01*len(images_test))):
        # printing the ASCII image
        # for row in range(28):
        #     row_str = ""
        #     for c in range(28):
        #         # row_str += str(list(images_test[(28*row)+c].int().data))
        #         row_str += str( images_test.int().tolist()[(28*row)+c][0] )
        #     print(row_str)

        cur_img = images_test[[i]]

        logits = model(cur_img)

        counts = logits.exp()

        prob = counts/counts.sum(1, keepdim=True)

        guess = list(list(prob)[0]).index(max(list(list(prob)[0])))
        print(guess, ans_test[i])
sample()


def calc_test_dataset_results():
    logits = model(images_test)

    loss = torch.nn.functional.cross_entropy(logits, ans_test)

    # Get predicted class (assuming y_pred contains raw logits or probabilities)
    y_pred_labels = torch.argmax(logits, dim=1)
        
    # Compare predictions with true labels
    correct = (y_pred_labels == ans_test).sum().item()
        
    # Compute accuracy
    accuracy = correct / ans_test.size(0)

    print("Test loss:")
    print(loss)

    print("Test accuracy:")
    print(accuracy)
calc_test_dataset_results()


def predict_external_image():
    cur_img = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).float()

    logits = model(cur_img)

    counts = logits.exp()

    prob = counts/counts.sum(1, keepdim=True)

    guess = list(list(prob)[0]).index(max(list(list(prob)[0])))
    print("External image guess: ", guess)
predict_external_image()


def save_prediction_to_csv():
    guesses = [
        ["ImageId", "Label"]
    ]
    for i in range(len(full_dataset)):
        cur_img = full_dataset[[i]]

        logits = model(cur_img)

        counts = logits.exp()

        prob = counts/counts.sum(1, keepdim=True)

        guess = list(list(prob)[0]).index(max(list(list(prob)[0])))

        guesses.append([i+1, guess])

    with open("output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(guesses)
save_prediction_to_csv()


for p in params:
    print(torch.argmax(p))
