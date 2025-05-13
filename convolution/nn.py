import os
import torch
import json
import torch.nn.functional as f

device = torch.device("cpu")

data = []
answers = []

# reading and processing data
dir = "./convolution/data/train-json"
for file in os.listdir(dir)[:20]:
    file_path = os.path.join(dir, file)

    with open(file_path, "r") as json_file:
        json_data = json.loads(json_file.read())
        data.append(json_data)
        answers.append(1 if file.split(".")[0] == "dog" else 0)

data = torch.tensor(data, dtype=torch.float32, device=device)
answers = torch.tensor(answers, dtype=torch.int64, device=device)

data = data.permute(0, 3, 1, 2)

g = torch.Generator(device=device).manual_seed(1)

# weights init
convo_w = torch.randn((64, 3, 3), requires_grad=True, generator=g, dtype=torch.float32, device=device) 
wo = torch.randn((192, 2), generator=g) / 192**0.5

params = [convo_w, wo]

def conv_layer(batch):
    output_tensors = []
    for i in range(convo_w.shape[0]):
        kernel = convo_w[i]
            
        # # Reshape kernel for conv2d: (out_channels, in_channels/groups, H, W)
        # # For RGB, we want to apply same kernel to all channels
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        kernel = kernel.repeat(3, 1, 1, 1)  # (3, 1, 3, 3)
            
        # Apply convolution with padding=1 to maintain size
        output = f.conv2d(batch, kernel, padding=1, groups=3)
            
        # Permute back to (H, W, C) format
        output = output.squeeze(0).permute(1, 2, 0)
        output = torch.nn.functional.relu(output)
        output_tensors.append(output)
    
    return torch.cat(tuple(output_tensors), dim=-1)

def global_pooling(input):
    return input.mean(dim=(0, 1))

batch_size = 16

for p in params:
    p.requires_grad = True

for _ in range(100):
    batch_indecies = torch.randint(0, data.shape[0], (batch_size,), generator=g)

    img = data[batch_indecies]
    ans = answers[batch_indecies]

    conv_h = conv_layer(img)
    pooling_h = global_pooling(conv_h)
    logits = pooling_h @ wo

    loss = torch.nn.functional.cross_entropy(logits, ans)
    print(loss)

    for p in params:
        p.grad = None

    loss.backward()

    learning_rate = 0.5 if loss.data > 1 else 0.05

    for p in params:
        p.data -= learning_rate * p.grad       
