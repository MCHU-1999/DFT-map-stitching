import imreg_dft as ird
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np


basedir = os.path.join('..', 'example_img')
# the TEMPLATE
im0 = np.array(Image.open(os.path.join(basedir, "sample1.png")).convert('L'))
# the image to be transformed
im1 = np.array(Image.open(os.path.join(basedir, "sample3.png")).convert('L'))
result = ird.similarity(im0, im1, numiter=3)

assert "timg" in result
# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt
    ird.imshow(im0, im1, result['timg'])
    plt.show()
