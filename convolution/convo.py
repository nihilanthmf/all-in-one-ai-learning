import torch.nn.functional as f
import numpy as np
from PIL import Image

def convolute(pix_mat, kernel):
    # Reshape and permute the image tensor to (C, H, W) format
    # Assuming pix_mat is (H, W, C) where C=3 for RGB
    img_tensor = pix_mat.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    # Reshape kernel for conv2d: (out_channels, in_channels/groups, H, W)
    # For RGB, we want to apply same kernel to all channels
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    kernel = kernel.repeat(3, 1, 1, 1)  # (3, 1, 3, 3)
    
    # Apply convolution with padding=1 to maintain size
    output = f.conv2d(img_tensor, kernel, padding=1, groups=3)
    
    # Permute back to (H, W, C) format
    output = output.squeeze(0).permute(1, 2, 0)
    
    # Show the image
    # with torch.no_grad():
    #     output_np = output.detach().cpu().numpy()
    #     output_np = np.clip(output_np, 0, 255).astype(np.uint8)
        
    #     # Create PIL Image
    #     output_image = Image.fromarray(output_np, 'RGB')
    #     output_image.show()
    #     output_image.save("./convolution/output.png")
    
    return output