from PIL import Image
import os
import json

dir = "./convolution/data/train"
index = 0
for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    
    img = Image.open(file_path)
    img = img.resize((404, 360))

    classification = file.split('.')[0]
    
    pixels = list(img.getdata())

    pix_mat = []
    for p in range(1, 360):
        pix_mat.append(pixels[(p-1) * 404:p * 404])

    with open(f"./convolution/data/train-json/{classification}.{index}.json", "w") as json_file:
        json.dump(pix_mat, json_file)
        
    index += 1
      