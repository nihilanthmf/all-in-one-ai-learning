import os
import torch

data = []
answers = []

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

# reading and processing data
dir = "./dogs-vs-cats/data/train-txt-blackwhite-101by90"
for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    
    with open(file_path) as f:
        data.append([int(x) for x in f.read().split()])
        answers.append(1 if file.split(".")[0] == "dog" else 0)
data = torch.tensor(data, dtype=torch.float32, device=device)
answers = torch.tensor(answers, dtype=torch.float32, device=device)

g = torch.Generator(device=device).manual_seed(1)
        
# weights init
wh = torch.randn((9090, 15000), generator=g, dtype=torch.float32, device=device) / 9090**0.5
bh = torch.zeros((15000), dtype=torch.float32, device=device) * 0

wo = torch.randn((15000, 2), generator=g, dtype=torch.float32, device=device)* 0.01
bo = torch.randn((2),  dtype=torch.float32, device=device) * 0

params = [wh, bh, wo, bo]

# function initialization
def model(input):
    hpreact = input @ wh + bh
    h = torch.tanh(hpreact)
    
    logits = h @ wo + bo
    
    return logits

for p in params:
    p.requires_grad = True

# training the network
batch_size = 100
for i in range(100000):
    # forward pass
    batch_indicies = list(torch.randint(0, len(data), (batch_size,), device=device))
    batch = data[batch_indicies]

    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, answers[batch_indicies])
    
    if i % 100 == 0:
        print(loss)
    
    # backward pass
    for p in params:
        p.grad = None
        
    loss.backward()
    
    learning_rate = 0.1
    for p in params:
        p.data -= learning_rate * p.grad
    