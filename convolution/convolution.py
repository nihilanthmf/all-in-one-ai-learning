from PIL import Image
import numpy as np

img = Image.open("./convolution/cat.jpg")
    
pixels = list(img.getdata())
# print(pixels)
size = list(img.size)

pix_mat = []
final_pix_mat = []
for p in range(1, size[1]):
    pix_mat.append(pixels[(p-1) * size[0]:p * size[0]])
    
kernel = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
]

# normalizing the kernel
normalized_kernel = []
kernel_sum = sum([sum([p for p in row]) for row in kernel])

for row_index in range(len(kernel)):
    for p_index in range(len(kernel[0])):
        kernel[row_index][p_index] /= kernel_sum


#
for row_index in range(1, len(pix_mat) - 1):
    final_pix_mat.append([])
    for pixel_index in range(1, len(pix_mat[row_index]) - 1):
        pixel_1 = pix_mat[row_index - 1][pixel_index - 1]
        pixel_2 = pix_mat[row_index][pixel_index - 1]
        pixel_3 = pix_mat[row_index + 1][pixel_index - 1]
        
        pixel_4 = pix_mat[row_index - 1][pixel_index]
        pixel_5 = pix_mat[row_index][pixel_index]
        pixel_6 = pix_mat[row_index + 1][pixel_index]
        
        pixel_7 = pix_mat[row_index - 1][pixel_index + 1]
        pixel_8 = pix_mat[row_index][pixel_index + 1]
        pixel_9 = pix_mat[row_index + 1][pixel_index + 1]
        
        near_pix_mat = [
            [pixel_1, pixel_2, pixel_3],
            [pixel_4, pixel_5, pixel_6],
            [pixel_7, pixel_8, pixel_9],
        ]
        
        final_pix_value = []
        for k_row_index in range(len(kernel)):
            for k_val_index in range(len(kernel[k_row_index])):
                cur_pix = list(near_pix_mat[k_row_index][k_val_index])
                
                final_pix_value = [kernel[k_row_index][k_val_index] * color for color in cur_pix]
                    
        final_pix_mat[-1].append(final_pix_value)


# drawing the final image
pixel_array = np.array(final_pix_mat, dtype=np.float32)

# Normalize pixel values to 0-255
pixel_array -= pixel_array.min()  # Shift min to 0
pixel_array /= pixel_array.max()  # Scale between 0-1
pixel_array *= 255

pixel_array = pixel_array.astype(np.uint8)  # Convert to uint8

# Create image
image = Image.fromarray(pixel_array, 'RGB')

# Show and save the image
image.show()
image.save("output.png")
