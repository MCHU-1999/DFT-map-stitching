import imreg_dft as ird
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from scipy.signal import convolve2d 

output_dir = "test_50px_filtered"
os.makedirs(output_dir, exist_ok=True)
basedir = os.path.join('.', 'maps_cropped_50px')

def blur(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    if image.ndim == 2:
        return convolve2d(image, kernel, mode='same', boundary='symm')
    elif image.ndim == 3:
        return np.stack([convolve2d(image[:, :, c], kernel, mode='same', boundary='symm') for c in range(image.shape[2])], axis=2)

def laplacian_edge(gray_image):
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]])
    edges = convolve2d(gray_image, kernel, mode='same', boundary='symm')
    edges = np.clip((edges - edges.min()) / (edges.max() - edges.min()) * 255, 0, 255)
    return edges

def upsample(image, scale_factor=2):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    # Create coordinate grids
    row_idx = np.linspace(0, h - 1, new_h)
    col_idx = np.linspace(0, w - 1, new_w)
    row_floor = np.floor(row_idx).astype(int)
    col_floor = np.floor(col_idx).astype(int)
    row_ceil = np.clip(row_floor + 1, 0, h - 1)
    col_ceil = np.clip(col_floor + 1, 0, w - 1)
    row_alpha = row_idx - row_floor
    col_alpha = col_idx - col_floor

    def interp_channel(channel):
        top = (1 - col_alpha)[None, :] * channel[row_floor[:, None], col_floor] + col_alpha[None, :] * channel[row_floor[:, None], col_ceil]
        bottom = (1 - col_alpha)[None, :] * channel[row_ceil[:, None], col_floor] + col_alpha[None, :] * channel[row_ceil[:, None], col_ceil]
        return (1 - row_alpha)[:, None] * top + row_alpha[:, None] * bottom

    if image.ndim == 2:
        return interp_channel(image)
    else:
        return np.stack([interp_channel(image[:, :, c]) for c in range(image.shape[2])], axis=2)

def rgb_to_grayscale(image):
    if image.ndim == 3 and image.shape[2] == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    elif image.ndim == 2:
        return image  # Already grayscale
    else:
        raise ValueError("Input must be an RGB image with 3 channels.")
    
def filter_by_percentile(image, percentile=75, binary=False):
    if image.ndim != 2:
        raise ValueError("Image must be 2D (grayscale).")

    threshold = np.percentile(image, percentile)

    if binary:
        output = np.where(image <= threshold, 0, 255)
    else:
        output = np.where(image <= threshold, image, 255)

    return output

def filter_by_value(image, min_val=0, max_val=255, fill_val=255):

    if image.ndim != 2:
        raise ValueError("Image must be 2D (grayscale).")

    mask = (image >= min_val) & (image <= max_val)
    result = np.where(mask, image, fill_val)

    return result.astype(np.uint8)

def quantize_to_bins(image, num_bins=10, output_range=(0, 255)):
    
    if image.ndim != 2:
        raise ValueError("Image must be 2D (grayscale).")

    # Flatten and get bin edges based on percentiles (uniform bins over intensity range)
    min_val, max_val = np.min(image), np.max(image)
    bin_edges = np.linspace(min_val, max_val + 1, num_bins + 1)

    # Digitize pixel values into bin indices (0 to num_bins - 1)
    bin_indices = np.digitize(image, bin_edges) - 1

    # Map bin index to output values
    output_min, output_max = output_range
    scaled_values = np.linspace(output_min, output_max, num_bins)

    quantized = scaled_values[bin_indices]
    return quantized

if __name__ == "__main__":
    for i in range(1, 11, 1):
        im0 = np.array(Image.open(os.path.join(basedir, f"a_{i:03d}.png")))
        im1 = np.array(Image.open(os.path.join(basedir, f"b_{i:03d}.png")))

        im0 = rgb_to_grayscale(im0)
        im1 = rgb_to_grayscale(im1)

        im0 = quantize_to_bins(im0, 10, (2, 255))
        im1 = quantize_to_bins(im1, 10, (2, 255))

        im0 = filter_by_percentile(im0, 5, False)
        im1 = filter_by_percentile(im1, 5, False)

        # im0 = upsample(im0, scale_factor=2)
        # im1 = upsample(im1, scale_factor=2)

        # im0 = blur(im0, kernel_size=3)
        # im1 = blur(im1, kernel_size=3)

        # im0 = laplacian_edge(im0)
        # im1 = laplacian_edge(im1)

        # the TEMPLATE
        # im0 = np.array(Image.open(os.path.join(basedir, f"a_{i:03d}.png")).convert('L'))
        # the image to be transformed
        # im1 = np.array(Image.open(os.path.join(basedir, f"b_{i:03d}.png")).convert('L'))
        result = ird.similarity(im0, im1, numiter=5)

        assert "timg" in result
        # Maybe we don't want to show plots all the time
        if os.environ.get("IMSHOW", "yes") == "yes":
            ird.imshow(im0, im1, result['timg'], cmap='gray')
            # plt.show()
            out_img = os.path.join(output_dir, f"test_pair_{i:03d}.png")
            plt.savefig(out_img)
