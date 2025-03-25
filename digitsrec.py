import torch

# reading the data
f = open("digits.txt")
content = f.readlines()[:15]
f.close()

images = []
ans = []
for img in content:
    imgspl = img.split(",")
    ans.append(int(imgspl[0]))
    images.append([int(x) for x in imgspl[1:]])

images = torch.tensor(images).float()
ans = torch.tensor(ans)

g = torch.Generator().manual_seed(2147483647)

w = torch.randn((784, 10), generator=g)
b = torch.randn((10), generator=g)

print(images.shape, w.shape)

def relu(n):
    return max(0, n)

for i in range(1):
    # logits = [[relu(x) for x in row] for row in img @ w + b]

    logits = torch.relu(images @ w+b)

    print(logits.shape)

    counts = logits.exp()

    prob = counts/counts.sum(1, keepdim=True)

    loss = -prob[torch.arange(784), ans].log().mean()

    print(loss)