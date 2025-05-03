import imreg_dft as ird
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np







if __name__ == "__main__":
    basedir = os.path.join('.', 'examples')

    # Read and convert images to grayscale, then to numpy arrays
    im0 = Image.open(os.path.join(basedir, "sample1.png")).convert('L')
    im1 = Image.open(os.path.join(basedir, "sample3.png")).convert('L')

    im0_array = np.array(im0)
    im1_array = np.array(im1)

    result = ird.similarity(im0_array, im1_array, numiter=3)

    assert "timg" in result
    if os.environ.get("IMSHOW", "yes") == "yes":
        ird.imshow(im0_array, im1_array, result['timg'])
        plt.show()
