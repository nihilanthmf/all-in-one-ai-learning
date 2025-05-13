import torch
import random
import matplotlib.pyplot as plt
import torch.nn.functional as f
import csv

# reading the data
file = open("./digits-recognition/digits.txt")
content = file.readlines()#[:1000]
random.shuffle(content)
file.close()

file = open("./digits-recognition/digits-test.txt")
content_kaggle = file.readlines()
file.close()

images = []
ans = []
images_test = []
ans_test = []


kaggleCompetitionDataset = []
# creating full dataset
for img in content_kaggle[:len(content_kaggle)]:
    imgspl = img.split(",")
    kaggleCompetitionDataset.append([int(x)/255 for x in imgspl])

# creating train dataset
for img in content[:int(0.95*len(content))]:
    imgspl = img.split(",")
    ans.append(int(imgspl[0]))
    images.append([int(x)/255 for x in imgspl[1:]])

# creating test dataset
for img in content[int(0.95*len(content)):]:
    imgspl = img.split(",")
    ans_test.append(int(imgspl[0]))
    images_test.append([int(x)/255 for x in imgspl[1:]])

images = torch.tensor(images).float()
ans = torch.tensor(ans)

images_test = torch.tensor(images_test).float()
ans_test = torch.tensor(ans_test)

kaggleCompetitionDataset = torch.tensor(kaggleCompetitionDataset).float()


# initializing the network's params
g = torch.Generator().manual_seed(1)

convo_w = torch.randn((32, 3, 3),  generator=g, dtype=torch.float32)
wh = torch.randn((196*32, 75), generator=g, dtype=torch.float32) / (196*32) **0.5
bh = torch.zeros((75,)) 

wo = torch.randn((75, 10), generator=g, dtype=torch.float32) / 75**0.5
bo = torch.zeros((10,), ) 

params = [convo_w, wh, bh, wo, bo]

for p in params:
    p.requires_grad=True

loss_values=[]

batch_size = 32


def convolute(pix_mat:torch.tensor, kernel):
    img_tensor = pix_mat.view(batch_size, 28, 28).unsqueeze(1)#.unsqueeze(0)#.repeat(16, 1, 1, 1)

    kernel = kernel.unsqueeze(1)#.unsqueeze(0)

    output = f.conv2d(input=img_tensor, weight=kernel, padding=1)
    
    output = output.squeeze(1)
    
    return output


def model(img):
    img = convolute(img, convo_w)#.view(batch_size, 784*32)

    img = f.max_pool2d(img, kernel_size=2, stride=2).view(batch_size, 196*32)

    h = torch.tanh(img @ wh + bh)
    
    logits = h @ wo + bo

    return logits


# forward + backward pass

# Test loss after proper (not really) weight initialization: 0.0781

# training loop
for i in range(21000):
    batch_indecies = torch.randint(0, images.shape[0], (batch_size,), generator=g)

    img = images[batch_indecies]
    curAns = ans[batch_indecies]

    logits = model(img)

    loss = f.cross_entropy(logits, curAns)

    if i % 100 == 0 and i > 3000:
        print(loss)
        loss_values.append(loss)

    for p in params:
        p.grad = None

    loss.backward()
    
    learningAlpha = 0.01

    if i > 5000:
        learningAlpha = 0.004
    # if i > 17000:
    #     learningAlpha = 0.00275

    for p in params:
        p.data -= learningAlpha * p.grad

# inference
with torch.no_grad():
    batch_size = 1
    for i in range(images_test.shape[0]//10):
        cur_img = images_test[i]
        ans = ans_test[i]

        logits = model(cur_img)

        counts = logits.exp()

        prob = counts/counts.sum(1, keepdim=True)

        guess = list(list(prob)[0]).index(max(list(list(prob)[0])))
        print(guess, ans)


# Calculating test loss/accuracy

    batch_size = images_test.shape[0]

    logits = model(images_test)

    loss = f.cross_entropy(logits, ans_test)

    y_pred_labels = torch.argmax(logits, dim=1)
        
    correct = (y_pred_labels == ans_test).sum().item()
        
    accuracy = correct / ans_test.size(0)

    print("Test loss:")
    print(loss)

    print("Test accuracy:")
    print(accuracy)



# saving predictions to svg for kaggle
with torch.no_grad():
    batch_size = 1

    guesses = [
        ["ImageId", "Label"]
    ]
    for i in range(len(kaggleCompetitionDataset)):
        cur_img = kaggleCompetitionDataset[[i]]

        logits = model(cur_img)

        counts = logits.exp()

        prob = counts/counts.sum(1, keepdim=True)

        guess = list(list(prob)[0]).index(max(list(list(prob)[0])))

        guesses.append([i+1, guess])

    with open("output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(guesses)


# plotting the loss
plt.plot(torch.tensor(loss_values), label="Training Loss", color='b', linestyle='-')
plt.show()
