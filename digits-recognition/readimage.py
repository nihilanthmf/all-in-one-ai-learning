from PIL import Image

im = Image.open('two.jpg')
pix_val = list(im.getdata())
print(pix_val)
res = []
for i in pix_val:
    if i > (200,200,200,200):
        res.append(0)
    else:
        res.append(1)
print(res)

# printing the image in ascii
for row in range(28):
    row_str = ""
    for c in range(28):
        row_str += str(res[(28*row)+c])

    print(row_str)