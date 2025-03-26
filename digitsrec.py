import torch

# reading the data
f = open("digits.txt")
content = f.readlines()
f.close()

images = []
ans = []
images_test = []
ans_test = []

# creating train dataset
for img in content[:int(0.8*len(content))]:
    imgspl = img.split(",")
    ans.append(int(imgspl[0]))
    images.append([1 if int(x)> 10 else 0 for x in imgspl[1:]])

# creating test dataset
for img in content[int(0.8*len(content)):]:
    imgspl = img.split(",")
    ans_test.append(int(imgspl[0]))
    images_test.append([1 if int(x)> 10 else 0 for x in imgspl[1:]])

images = torch.tensor(images).float()
ans = torch.tensor(ans)

images_test = torch.tensor(images_test).float()
ans_test = torch.tensor(ans_test)

g = torch.Generator().manual_seed(2147483647)

w = torch.randn((784, 10), generator=g)
b = torch.randn((10), generator=g)
params = [w, b]

for p in params:
    p.requires_grad = True

# training dataset
miniBatchCount = 200
for i in range(100000):
    minibatch = torch.randint(0, images.shape[0], (miniBatchCount,), generator=g)

    logits = torch.tanh(images[minibatch] @ w+b)

    counts = logits.exp()

    prob = counts/counts.sum(1, keepdim=True)

    loss = -prob[torch.arange(prob.shape[0]), ans[minibatch]].log().mean()

    if i % 100 == 0:
        print(loss)

    for p in params:
        p.grad = None

    loss.backward()

    learningAlpha = 0.15 if i < 1000 else 0.04
    for p in params:
        p.data -= learningAlpha * p.grad

# sampling
print("Guesses:")
for i in range(len(images_test)):
    cur_img = images_test[[i]]
    logits = torch.tanh(cur_img @ w+b)

    counts = logits.exp()

    prob = counts/counts.sum(1, keepdim=True)

    guess = list(list(prob)[0]).index(max(list(list(prob)[0])))
    print(guess, ans_test[i])

# calculating test loss
logits = torch.tanh(images_test @ w+b)

counts = logits.exp()

prob = counts/counts.sum(1, keepdim=True)

loss = -prob[torch.arange(prob.shape[0]), ans_test].log().mean()
print("Test loss:")
print(loss)