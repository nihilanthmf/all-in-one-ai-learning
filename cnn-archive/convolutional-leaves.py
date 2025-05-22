import torch
import random
import matplotlib.pyplot as plt
import torch.nn.functional as f
import csv
import os
import json
from PIL import Image

# reading the data
file = open("./leaf-classification/train.csv")
content = file.readlines()[1:]#[:500]
file.close()

file = open("./leaf-classification/test.csv")
content_kaggle = file.readlines()[1:]
file.close()

images = []
ans = []
images_test = []
ans_test = []

content_train = content[:int(0.95*len(content))]
content_test = content[int(0.95*len(content)):]

# random.shuffle(content_train)
kaggleCompetitionDataset = []
# creating full dataset
for img in content_kaggle[:len(content_kaggle)]:
    imgspl = img.split(",", 1)
    kaggleCompetitionDataset.append([int(x)/255 for x in imgspl])

# creating train dataset
for img in content_train:
    imgspl = img.split(",")
    ans.append(imgspl[1])
    images.append([float(x) for x in imgspl[2:]])

# creating test dataset
for img in content_test:
    imgspl = img.split(",")
    ans_test.append(int(imgspl[1]))
    images_test.append([float(x) for x in imgspl[2:]])

images = torch.tensor(images).float()
ans = torch.tensor(ans)

images_test = torch.tensor(images_test).float()
ans_test = torch.tensor(ans_test)

kaggleCompetitionDataset = torch.tensor(kaggleCompetitionDataset).float()

# initializing the network's params
g = torch.Generator().manual_seed(1)

batch_size = 4 # will be 2x that cause both dogs and cats
convo_w_kernels = 4

weights_scale = 0.075

convo_w = torch.randn((convo_w_kernels, 3, 3, 3), generator=g, dtype=torch.float32) * weights_scale

wh = torch.randn((201216, 50), generator=g, dtype=torch.float32) * weights_scale
bh = torch.zeros((50,)) 

wo = torch.randn((50, 10), generator=g, dtype=torch.float32) * weights_scale
bo = torch.zeros((10,), )

params = [convo_w, wh, bh, wo, bo]

for p in params:
    p.requires_grad=True

loss_values=[]

def saveParams():
    with open(f"./weights/convo_w.txt", "w") as f:
        json.dump(convo_w.tolist(), f)
    with open(f"./weights/wh.txt", "w") as f:
        json.dump(wh.tolist(), f)
    with open(f"./weights/bh.txt", "w") as f:
        json.dump(bh.tolist(), f)
    with open(f"./weights/wo.txt", "w") as f:
        json.dump(wo.tolist(), f)
    with open(f"./weights/bo.txt", "w") as f:
        json.dump(bo.tolist(), f)

def convolute(img_tensor:torch.tensor, kernel):
    output = f.conv2d(input=img_tensor, weight=kernel, padding=1)
    
    output = output.squeeze(1)
    
    return output


def model(img):
    img_convo = convolute(img, convo_w)

    img_activations = torch.relu(img_convo)

    img_pooling = f.max_pool2d(img_activations, kernel_size=2, stride=2).view(batch_size*2, -1)

    h = torch.tanh(img_pooling @ wh + bh)
    
    logits = h @ wo + bo

    return logits

sum_loss = 0

def readImageBatch(indecies):
    images = []
    ans = []

    # 0 = cat; 1 = dog
    def readImg(cat, i):
        img = Image.open(os.path.join(dir, f"cat.{i}.jpg" if cat else f"dog.{i}.jpg"))
        img = img.resize((img.size[0]//2, img.size[1]//2))
        width, height = img.size

        pad_w = 525 - width
        pad_h = 384 - height

        pad = (pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2)

        pixels = [[img.getpixel((x, y)) for x in range(width)] for y in range(height)]
        pixels = torch.tensor(pixels).permute(2,0,1)

        padded_img = f.pad(pixels, pad, mode='constant', value=0)
        
        images.append(padded_img.tolist())
        ans.append(0 if cat else 1)

    for index in indecies:
        readImg(True, index)
        readImg(False, index)

    return [images, ans]

# training loop
for i in range(10000):
    randomIndecies = [random.randint(1, 9115) for _ in range(batch_size)]

    readImagesRaw = readImageBatch(randomIndecies)
    img = torch.tensor(readImagesRaw[0]).float()
    ans = torch.tensor(readImagesRaw[1])

    logits = model(img)

    # calc loss
    loss = f.cross_entropy(logits, ans)

    # applying L2 regularization
    loss += 0.000035 * sum(p.pow(2).sum() for p in params)
    
    loss_values.append(sum_loss / 10)
    sum_loss = 0
    print(f"L2 loss epoch {i}: {loss.data}")
    sum_loss += loss.item()

    for p in params:
        p.grad = None

    loss.backward()
    
    learningAlpha = 0.05

    if i > 5000:
        learningAlpha = 0.01

    for p in params:
        p.data -= learningAlpha * p.grad

saveParams()

# # Calculating test loss/accuracy

#     batch_size = images_test.shape[0]

#     logits = model(images_test)

#     loss = f.cross_entropy(logits, ans_test)

#     y_pred_labels = torch.argmax(logits, dim=1)
        
#     correct = (y_pred_labels == ans_test).sum().item()
        
#     accuracy = correct / ans_test.size(0)

#     print("Test loss:")
#     print(loss)

#     print("Test accuracy:")
#     print(accuracy)