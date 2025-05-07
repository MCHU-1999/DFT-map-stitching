import imreg_dft as ird
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

output_dir = "test_30px"
os.makedirs(output_dir, exist_ok=True)
basedir = os.path.join('.', 'maps_cropped_50px')

for i in range(1, 11, 1):
    # the TEMPLATE
    im0 = np.array(Image.open(os.path.join(basedir, f"a_{i:03d}.png")).convert('L'))
    # the image to be transformed
    im1 = np.array(Image.open(os.path.join(basedir, f"b_{i:03d}.png")).convert('L'))
    result = ird.similarity(im0, im1, numiter=5)

    assert "timg" in result
    # Maybe we don't want to show plots all the time
    if os.environ.get("IMSHOW", "yes") == "yes":
        ird.imshow(im0, im1, result['timg'], cmap='gray')
        # plt.show()
        out_img = os.path.join(output_dir, f"test_pair_{i:03d}.png")
        plt.savefig(out_img)
