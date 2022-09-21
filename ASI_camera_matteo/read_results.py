from astropy.io import fits as pf
from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.insert(0, '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/libs')
import utils as al

#image to read path

file_path = '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/Immagini/Risultati/result_analysis.fits'

histo_path = '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/Immagini/Risultati/saved_histo.npz'

al.retrive_histo(histo_path)

data_f = pf.open(file_path, memmap = True)
data_f.info()   #mostra a schermo info file

image_data = pf.getdata(file_path, ext = 0)/4.
print(image_data.shape)
flat_image = image_data.flatten()

plt.figure()
plt.imshow(image_data, cmap = 'plasma')
plt.colorbar()
plt.show()
