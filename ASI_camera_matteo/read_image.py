
from astropy.io import fits as pf
from matplotlib import pyplot as plt
import numpy as np

#image to read path

file_path = '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/Immagini/Risultati/result_analysis.fits'




data_f = pf.open(file_path, memmap = True)
data_f.info()   #mostra a schermo info file

image_data = pf.getdata(file_path, ext = 0)/4.
print(image_data.shape)
flat_image = image_data.flatten()


fig, ax = plt.subplots()
ax.hist(flat_image, bins = int(65536/4), range = (0,65536/4)   , alpha = 1, histtype = 'step')
mean = flat_image.mean()
rms = flat_image.std()
s = 'mean = ' + str(round(mean,3)) + "\n" + "RMS = " + str(round(rms,3))
ax.text(0.7, 0.9, s,  transform = ax.transAxes,  bbox = dict(alpha=0.7))


plt.figure()
plt.imshow(image_data, cmap = 'plasma')
plt.colorbar()
plt.show()


