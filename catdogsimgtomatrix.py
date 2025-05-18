from PIL import Image
import os
import json

dir = "./dogs-vs-cats-redux-kernels-edition/train"
index = 0
for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    
    img = Image.open(file_path)
    # img = img.convert('RGB')
    # img = img.resize((404, 360))

    classification = file.split('.')[0]
    
    width, height = img.size

    pixels = [[img.getpixel((x, y)) for x in range(width)] for y in range(height)]

    with open(f"./dogs-vs-cats-redux-kernels-edition/train-json/{classification}.{index}.json", "w") as json_file:
        json.dump(pixels, json_file)
        
    index += 1
    print(index)
    