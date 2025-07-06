import os
import numpy as np
from scipy.ndimage import rotate, shift, convolve, laplace, gaussian_filter
from scipy.signal.windows import tukey, hann

def pad_bg_value(image, size):
    # Ensure size is a tuple (new_h, new_w)
    if isinstance(size, int):
        new_h = new_w = size
    else:
        new_h, new_w = size

    h, w = image.shape
    if new_h < h or new_w < w:
        raise ValueError(f"Target size ({new_h}, {new_w}) is smaller than image size ({h}, {w}).")

    # Compute total padding needed in each dimension
    pad_h_total = new_h - h
    pad_w_total = new_w - w
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    # fill_value = np.percentile(image, 99)
    fill_value = 0

    # Apply symmetric padding with a constant fill value
    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=fill_value
    )
    return padded

def rgb_to_grayscale(image):
    if image.ndim == 3 and image.shape[2] == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    elif image.ndim == 2:
        return image  # Already grayscale
    else:
        raise ValueError("Input must be an RGB image with 3 channels.")
    
def negative_film(image):
    return 255 - image

def bg_subtr_global(im0, im1):
    d0 = min(np.min(im0), np.min(im1))
    return im0 - d0, im1 - d0

def bg_subtr(im0):
    return im0 - np.min(im0)

def contrast_stretching(image, low_percentile=1, high_percentile=99):
    """
    Contrast stretching algorithm that maps percentile values to [0,1] then scales to [0,255].
    Equivalent to MATLAB's imadjust() with default settings.
    """
    # Calculate percentile values
    p_low = np.percentile(image, low_percentile)
    p_high = np.percentile(image, high_percentile)
    
    # Linear mapping: map [p_low, p_high] to [0, 1]
    stretched = (image - p_low) / (p_high - p_low)
    
    # Clip values to [0, 1] range
    stretched = np.clip(stretched, 0, 1)
    
    # Scale to [0, 255] range
    enhanced = stretched * 255
    
    return enhanced

def filter_by_percentile(image, percentile=99):
    """
    Filter image by percentile threshold - set all values below percentile to 0.
    """    
    # Calculate percentile threshold value
    threshold_value = np.percentile(image, percentile)
    
    # Create binary mask: True where pixels >= threshold
    mask = image >= threshold_value
    
    # Apply filter: keep original values above threshold, set others to 0
    filtered = np.where(mask, image, 0)

    return filtered

def apply_hann_window(img):
    """
    Apply Hann windowing directly to an image.
    """
    # Create and apply Hann window
    win_y = hann(img.shape[0])
    win_x = hann(img.shape[1])
    window = np.outer(win_y, win_x)
    
    return img * window

def apply_tukey_window(img, alpha=0.2):
    """
    Apply Tukey windowing directly to an image.
    """    
    # Create and apply Tukey window
    win_y = tukey(img.shape[0], alpha)
    win_x = tukey(img.shape[1], alpha)
    window = np.outer(win_y, win_x)
    
    return img * window


def ransac_align(im1, im2, iteration_num=100, max_translation=10, bgval=None):
    """
    Align im2 to im1 using rotation and translation with constant bg padding.

    Args:
        im1 (ndarray): Fixed image (50x50).
        im2 (ndarray): Moving image (50x50).
        iteration_num (int): Number of RANSAC iterations.
        max_translation (int): Maximum pixel shift in x and y.
        bgval (float, optional): Background fill value. If None, uses 99th percentile of im2.

    Returns:
        aligned_im2 (ndarray): Transformed version of im2.
        best_params (tuple): (rotation_deg, shift_y, shift_x)
    """
    assert im1.shape == (50, 50) and im2.shape == (50, 50), "Images must be 50x50"

    if bgval is None:
        bgval = np.percentile(im2, 99)

    best_score = float('inf')
    best_transformed = None
    best_params = None

    pad_width = max_translation + 5
    im2_padded = np.pad(im2, pad_width, mode='constant', constant_values=bgval)

    for _ in range(iteration_num):
        angle = np.random.uniform(0, 360)
        tx = np.random.uniform(-max_translation, max_translation)
        ty = np.random.uniform(-max_translation, max_translation)

        rotated = rotate(im2_padded, angle, reshape=False, order=0, mode='constant', cval=bgval)
        transformed = shift(rotated, shift=(ty, tx), order=0, mode='constant', cval=bgval)

        center_y, center_x = transformed.shape[0] // 2, transformed.shape[1] // 2
        cropped = transformed[center_y - 25:center_y + 25, center_x - 25:center_x + 25]

        diff = np.abs(im1 - cropped)
        score = np.sum(diff)

        if score < best_score:
            best_score = score
            best_transformed = cropped.copy()
            best_params = (angle, ty, tx)

    return best_transformed, best_params
