import torch
import random
import matplotlib.pyplot as plt
import torch.nn.functional as f
import csv
import os
import json
from PIL import Image
import numpy as np

dir = "./dogs-vs-cats-redux-kernels-edition/train"

device = "cpu"

# initializing the network's params
g = torch.Generator(device=device).manual_seed(42)

batch_size = 32 # will be 2x that cause both dogs and cats
convo_w_kernels = 128

weights_scale = 0.001

convo_w = torch.randn((convo_w_kernels, 3, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w2 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w3 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w4 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale

convo_w5 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w6 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w7 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w8 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale

wh = torch.randn((8192, 10), generator=g, dtype=torch.float32, device=device) * weights_scale
bh = torch.zeros((10,), device=device)

wo = torch.randn((10, 2), generator=g, dtype=torch.float32, device=device) * weights_scale
bo = torch.zeros((2,), device=device)


convo_w = convo_w.to(device)
wh = wh.to(device)
bh = bh.to(device)
wo = wo.to(device)
bo = bo.to(device)

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

def model(img, epoch):
    img_convo = convolute(img, convo_w)
    img_activations = f.relu(img_convo)
    img_convo_2 = convolute(img_activations, convo_w2)
    img_activations_2 = f.relu(img_convo_2)

    img_pooling = f.max_pool2d(img_activations_2, kernel_size=2, stride=2)

    img_convo_3 = convolute(img_pooling, convo_w3)
    img_activations_3 = f.relu(img_convo_3)
    img_convo_4 = convolute(img_activations_3, convo_w4)
    img_activations_4 = f.relu(img_convo_4)

    img_pooling_2 = f.max_pool2d(img_activations_4, kernel_size=2, stride=2)

    img_convo_5 = convolute(img_pooling_2, convo_w5)
    img_activations_5 = f.relu(img_convo_5)
    img_convo_6 = convolute(img_activations_5, convo_w6)
    img_activations_6 = f.relu(img_convo_6)

    img_pooling_3 = f.max_pool2d(img_activations_6, kernel_size=2, stride=2)

    img_convo_7 = convolute(img_pooling_3, convo_w7)
    img_activations_7 = f.relu(img_convo_7)
    img_convo_8 = convolute(img_activations_7, convo_w8)
    img_activations_8 = f.relu(img_convo_8)

    img_pooling_3 = f.max_pool2d(img_activations_8, kernel_size=2, stride=2)

    print(img_pooling_3.shape)

    flat = img_pooling_3.reshape(batch_size*2, -1)
    h = torch.tanh(flat @ wh + bh)
    
    logits = h @ wo + bo

    # h_detached = h.detach().abs()
    # plt.imshow(h_detached < 0.01, cmap="gray")    
    # plt.show()
    # plt.imshow(h_detached > 0.99, cmap="gray")    
    # plt.show()

    return logits

sum_loss = 0

def readImageBatch(indecies):
    images = []
    ans = []

    # 0 = cat; 1 = dog
    def readImg(cat, i):
        img = Image.open(os.path.join(dir, f"cat.{i}.jpg" if cat else f"dog.{i}.jpg"))
        img = img.resize((128, 128))
        
        pixels = np.array(img)
        
        images.append(pixels)
        ans.append(0 if cat else 1)

    for index in indecies:
        readImg(True, index)
        readImg(False, index)

    return [images, ans]

# training loop
for i in range(10000):
    randomIndecies = [random.randint(1, 9115) for _ in range(batch_size)]

    readImagesRaw = readImageBatch(randomIndecies)
    # img = torch.tensor(readImagesRaw[0], device=device).float().permute(0, 3, 1, 2) # that shit was 10x slower for some reason
    img = torch.from_numpy(np.array(readImagesRaw[0])).float().permute(0, 3, 1, 2).to(device) # that shit 10x speed for some fucking reason
    ans = torch.tensor(readImagesRaw[1], device=device).to(device)

    logits = model(img, i)

    # calc loss
    loss = f.cross_entropy(logits, ans)

    # applying L2 regularization
    loss += 0.000015 * sum(p.pow(2).sum() for p in params)

    sum_loss += loss
    if i % 10 == 0:
        loss_values.append(sum_loss / 10)
        sum_loss = 0

    print(f"{i} epoch L2 loss: {loss.data}")
    sum_loss += loss.item() 

    for p in params:
        p.grad = None

    loss.backward()
    
    learningAlpha = 0.1

    if i > 1000:
        learningAlpha = 0.005
    elif i > 200:
        learningAlpha = 0.05

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