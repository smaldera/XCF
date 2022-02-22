from astropy.io import fits as pf
from matplotlib import pyplot as plt

import numpy as np
import os

from sklearn import metrics

i = 0
flat_image = []


def write_fitsImage(array, nomefile):
   hdu = pf.PrimaryHDU(array)
   hdu.writeto(nomefile)


files_path = ("C:\\Users\\Acer\\Downloads\\Uni\\Tesi\\Dati\\solo dati FIT - no sorgente")

for filename in os.listdir(files_path):
    files = os.path.join(files_path, filename)
    data_f = pf.open(files, memmap = True)    
#    data_f.info()
    
    image_data = pf.getdata(files, ext=0)/4
    flat_image.insert(i, image_data.flatten())
    i = i+1

    
pixels_weights = np.array(flat_image)
pixels_weights.reshape(pixels_weights.shape[0], pixels_weights.shape[1])

#print(pixels_weights.shape)

rumor_mean = np.mean(pixels_weights, axis=0)
rumor_std = np.std(pixels_weights, axis=0)

print(rumor_mean)
print(rumor_std)

        
fig, ax = plt.subplots()
ax.hist(rumor_mean, bins=int(65536/4), range=(0,65536/4)   , alpha=1, histtype='step')
mean = rumor_mean.mean()
rms = rumor_mean.std()
s='mean='+str(round(mean,3))+"\n"+"RMS="+str(round(rms,3))
ax.text(0.7, 0.9, s,  transform=ax.transAxes,  bbox=dict(alpha=0.7))


a = np.array(rumor_mean.reshape(2822, 4144))

b = np.array(rumor_std.reshape(2822, 4144))


plt.figure()
plt.imshow(a, cmap='plasma')
plt.colorbar()
plt.show()


write_fitsImage(a, "meanped.fits")
write_fitsImage(b, "stdped.fits")
