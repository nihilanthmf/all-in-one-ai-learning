from PIL import Image
import os
import json

dir = "./leaf-classification/train.csv"
index = 0

maxWidth = 0
maxHeight = 0
for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    
    img = Image.open(file_path)
    # img = img.convert('RGB')
    # img = img.resize((404, 360))

    classification = file.split('.')[0]
    
    width, height = img.size

    maxWidth = max(maxWidth, width)
    maxHeight = max(maxHeight, height)

    # pixels = [[img.getpixel((x, y)) for x in range(width)] for y in range(height)]

    # with open(f"./dogs-vs-cats-redux-kernels-edition/train-json/{classification}.{index}.txt", "w") as f:
    #     json.dump(pixels, f)
        
    # index += 1
    # print(index)

print(maxWidth)#1050
print(maxHeight)#768


    