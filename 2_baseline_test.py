import imreg_dft as ird
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from pre_proc import rgb_to_grayscale, negative_film, bg_subtr, contrast_stretching, denoise_blur, apply_tukey_window, pad_bg_value, center_crop


basedir = os.path.join('./data_test', '100px')
output_dir = "./result_test/100px_window07"
os.makedirs(output_dir, exist_ok=True)


def calculate_image_difference(im2, im2_base):
    diff = np.abs(im2.astype(np.float64) - im2_base.astype(np.float64))
    return np.mean(diff)


if __name__ == "__main__":
    
    total_diff = 0.0 

    for i in range(1, 11, 1):
        im0 = np.array(Image.open(os.path.join(basedir, f"a_{i:03d}.png")))
        im1 = np.array(Image.open(os.path.join(basedir, f"b_{i:03d}.png")))

        im0 = rgb_to_grayscale(im0)
        im1 = rgb_to_grayscale(im1)

        im0 = negative_film(im0)
        im1 = negative_film(im1)

        im0 = bg_subtr(im0)
        im1 = bg_subtr(im1)

        im0_base = contrast_stretching(im0, 50, 95)
        im1_base = contrast_stretching(im1, 50, 95)

        # im0 = denoise_blur(im0_base)
        # im1 = denoise_blur(im1_base)

        im0 = apply_tukey_window(im0, 0.7)
        im1 = apply_tukey_window(im1, 0.7)

        # im0 = pad_bg_value(im0, 300)
        # im1 = pad_bg_value(im1, 300)

        result_base = ird.similarity(im0_base, im1_base, numiter=3)
        result = ird.similarity(im0, im1, numiter=3)
        im2_base = result_base['timg']
        im2 = result['timg']
        
        # im0 = center_crop(im0, 100)
        # im1 = center_crop(im1, 100)
        # im2 = center_crop(im2, 100)

        # Compute difference image with normalization
        norm = np.percentile(np.abs(im2_base), 99.5) / np.percentile(np.abs(im0_base), 99.5)
        phase_norm = np.median(np.angle(im2_base / im0_base) % (2 * np.pi))
        if phase_norm != 0:
            norm *= np.exp(1j * phase_norm)
        im3_base = abs(im2_base - im0_base * norm)

        # Maybe we don't want to show plots all the time
        if os.environ.get("IMSHOW", "yes") == "yes":
            ird.imshow(im0, im1, im2, cmap='gray', title=f"Pair {i}", subtitle=True, compare_img=im3_base)
            # plt.show()
            out_img = os.path.join(output_dir, f"test_pair_{i:03d}.png")
            plt.savefig(out_img)

        diff = calculate_image_difference(im2, im2_base)
        total_diff += diff
        print(f"Pair {i}: difference = {diff:.4f}")

    # print(f"Total difference sum: {total_diff:.4f}")
    print(f"Per pixel difference: {total_diff/10:.4f}")