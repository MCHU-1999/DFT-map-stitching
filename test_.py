import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def prepare_image(path):
    return np.array(Image.open(path).convert('L'))

def low_high_pass_filters(img, radius=30):
    # Perform FFT and shift zero-freq to center
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Create low-pass and high-pass masks
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2

    # Low-pass mask (centered circle)
    low_pass_mask = np.zeros_like(img)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
    low_pass_mask[mask_area] = 1

    # High-pass mask = inverse of low-pass
    high_pass_mask = 1 - low_pass_mask

    # Apply masks
    low_pass = fshift * low_pass_mask
    high_pass = fshift * high_pass_mask

    # Inverse FFT to get images back
    low_img = np.fft.ifft2(np.fft.ifftshift(low_pass)).real
    high_img = np.fft.ifft2(np.fft.ifftshift(high_pass)).real

    return low_img, high_img

if __name__ == "__main__":
    img_a = prepare_image('./map/a_2.jpg')
    img_b = prepare_image('./map/b_2.jpg')

    low_img_a, high_img_a = low_high_pass_filters(img_a, radius=30)
    low_img_b, high_img_b = low_high_pass_filters(img_b, radius=30)

    # Show the results
    plt.figure(figsize=(12, 6))

    plt.subplot(231)
    plt.imshow(img_a, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(low_img_a, cmap='viridis')
    plt.title('Low-Pass (Blurred)')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(high_img_a, cmap='viridis')
    plt.title('High-Pass (Edges)')
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(img_b, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(low_img_b, cmap='viridis')
    plt.title('Low-Pass (Blurred)')
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(high_img_b, cmap='viridis')
    plt.title('High-Pass (Edges)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

