import convo
import os
import torch
import json

device = torch.device("cpu")

data = []
answers = []

# reading and processing data
dir = "./convolution/data/train-json"
for file in os.listdir(dir)[:1]:
    file_path = os.path.join(dir, file)

    with open(file_path, "r") as json_file:
        json_data = json.loads(json_file.read())
        data.append(json_data)
        answers.append(1 if file.split(".")[0] == "dog" else 0)

data = torch.tensor(data, dtype=torch.float32, device=device)
answers = torch.tensor(answers, dtype=torch.float32, device=device)

g = torch.Generator(device=device).manual_seed(1)

# weights init
convo_w = torch.randn((3, 3, 64), generator=g, dtype=torch.float32, device=device)

kernel = [
    [100, 100, 100],
    [100, 100, 100],
    [100, 100, 100]
]
kernel_sum = sum([sum([p for p in row]) for row in kernel])

for row_index in range(len(kernel)):
    for p_index in range(len(kernel[0])):
        kernel[row_index][p_index] /= kernel_sum


convo.convolute(data[0], 
torch.tensor(kernel, dtype=torch.float32, requires_grad=True)
)

def conv_layer(batch):
    res = []
    for kernel in convo_w:
        for img in batch:   
            res.append(convo.convolute(img, kernel))

    return res

# def relu(batch):