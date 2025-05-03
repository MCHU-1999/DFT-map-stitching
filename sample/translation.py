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
result = ird.translation(im0, im1)
tvec = result["tvec"].round(4)

# the Transformed IMaGe.
timg = ird.transform_img(im1, tvec=tvec)

# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt
    ird.imshow(im0, im1, timg)
    plt.show()

print("Translation is {}, success rate {:.4g}"
      .format(tuple(tvec), result["success"]))
