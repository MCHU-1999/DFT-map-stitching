import imreg_dft as ird
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from pre_proc import rgb_to_grayscale, negative_film, bg_subtr, contrast_stretching, apply_tukey_window, pad_bg_value


basedir = os.path.join('.', 'maps_cropped_100px')
output_dir = "test_100px"
os.makedirs(output_dir, exist_ok=True)


if __name__ == "__main__":
    for i in range(1, 11, 1):
        im0 = np.array(Image.open(os.path.join(basedir, f"a_{i:03d}.png")))
        im1 = np.array(Image.open(os.path.join(basedir, f"b_{i:03d}.png")))

        im0 = rgb_to_grayscale(im0)
        im1 = rgb_to_grayscale(im1)

        im0 = negative_film(im0)
        im1 = negative_film(im1)

        im0 = bg_subtr(im0)
        im1 = bg_subtr(im1)

        im0 = contrast_stretching(im0, 50, 95)
        im1 = contrast_stretching(im1, 50, 95)

        # im0 = apply_tukey_window(im0)
        # im1 = apply_tukey_window(im1)

        # im0 = pad_bg_value(im0, 100)
        # im1 = pad_bg_value(im1, 100)

        result = ird.similarity(im0, im1, numiter=3)
        assert "timg" in result
        im2 = result['timg']

        # im2, params = ransac_align(im0, im1, iteration_num=1200)

        # Maybe we don't want to show plots all the time
        if os.environ.get("IMSHOW", "yes") == "yes":
            ird.imshow(im0, im1, im2, cmap='gray', title=f"Pair {i}", subtitle=True)
            # plt.show()
            out_img = os.path.join(output_dir, f"test_pair_{i:03d}.png")
            plt.savefig(out_img)
