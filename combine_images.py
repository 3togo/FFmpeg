from PIL import Image
import numpy as np
imgs = ['1.bmp', '1_dep.bmp', '1z.jpg']
concatenated = Image.fromarray(
  np.concatenate(
    [np.array(Image.open(x)) for x in imgs],
    axis=1
  )
)

