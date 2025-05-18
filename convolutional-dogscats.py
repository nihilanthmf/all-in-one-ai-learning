import torch
import random
import matplotlib.pyplot as plt
import torch.nn.functional as f
import csv
import os
import json

dir = "./dogs-vs-cats-redux-kernels-edition/train-json"

# initializing the network's params
g = torch.Generator().manual_seed(1)

batch_size = 8 # will be 2x that cause both dogs and cats
convo_w_kernels = 4

weights_scale = 0.075

convo_w = torch.randn((convo_w_kernels, 1, 3, 3), generator=g, dtype=torch.float32) * weights_scale
# convo_w2 = torch.randn((convo_w_kernels**2, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32) * weights_scale

wh = torch.randn((convo_w_kernels*14*14, 100), generator=g, dtype=torch.float32) * weights_scale
bh = torch.zeros((100,)) 

wo = torch.randn((100, 10), generator=g, dtype=torch.float32) * weights_scale
bo = torch.zeros((10,), )

params = [convo_w, wh, bh, wo, bo]

for p in params:
    p.requires_grad=True

loss_values=[]

def convolute(img_tensor:torch.tensor, kernel):
    output = f.conv2d(input=img_tensor, weight=kernel, padding=1)
    
    output = output.squeeze(1)
    
    return output


def model(img):
    img_tensor = img.view(batch_size, 28, 28).unsqueeze(1)

    img_convo = convolute(img_tensor, convo_w)

    img_activations = torch.relu(img_convo)

    img_pooling = f.adaptive_avg_pool2d(img_activations, (64, 64)).view(batch_size, -1)

    print(img_pooling.shape)

    h = torch.tanh(img_pooling @ wh + bh)
    
    logits = h @ wo + bo

    return logits

sum_loss = 0

def readImageBatch(indecies):
    # 0 = cat; 1 = dog
    images = []
    ans = []
    for i in indecies:
        images.append(json.loads(open(os.path.join(dir, f"cat.{i}.json"))))
        ans.append(0)
        images.append(json.loads(open(os.path.join(dir, f"dog.{i}.json"))))
        ans.append(1)

    return [images, ans]

# training loop
for i in range(200000):
    randomIndecies = [random.randint(1, 9115) for _ in range(16)]

    readImagesRaw = readImageBatch(randomIndecies)
    img = torch.tensor(readImagesRaw[0]).float()
    ans = torch.tensor(readImagesRaw[1])

    logits = model(img)

    # calc loss
    loss = f.cross_entropy(logits, ans)
    # applying L2 regularization
    loss += 0.000035 * sum(p.pow(2).sum() for p in params)
    

    if i % 10 == 0:
        loss_values.append(sum_loss / 10)
        sum_loss = 0
        print(f"L2 loss epoch {i}: {loss.data}")
    else:
        sum_loss += loss.item()

    for p in params:
        p.grad = None

    loss.backward()
    
    learningAlpha = 0.05

    if i > 5000:
        learningAlpha = 0.01

    for p in params:
        p.data -= learningAlpha * p.grad


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