from PIL import Image
import os
import json

# mean width 404.09904
# mean height 360.47808

# dir = "./dogs-vs-cats/data/train"
# for file in os.listdir(dir):
#     file_path = os.path.join(dir, file)
#     im = Image.open(file_path)
#     widths += im.size[0]
#     heights += im.size[1]
#     count += 1
    
# print(widths / count)
# print(heights / count)


dir = "./dogs-vs-cats/data/train"
index = 0
for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    
    img = Image.open(file_path)
    img = img.convert("L")
    img = img.resize((404//4, 360//4))

    classification = file.split('.')[0]
    
    pixels = list(img.getdata())
    with open(f"./dogs-vs-cats/data/train-txt-blackwhite-101by90/{classification}.{index}.txt", "w") as file:
        file.writelines(f"{item}\n" for item in pixels)
        
    index += 1
      