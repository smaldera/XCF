from astropy.io import fits as pf
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import os 
import math



i = 0
a = np.zeros(11694368)
b = np.zeros(11694368)


def write_fitsImage(array, nomefile):
   hdu = pf.PrimaryHDU(array)
   hdu.writeto(nomefile)


files_path = ("C:\\Users\\Acer\\Downloads\\Uni\\Tesi\\Dati\\acquisizione lunga")

for filename in os.listdir(files_path):
    files = os.path.join(files_path, filename)
    data_f = pf.open(files, memmap = True)    
    
    image_data = pf.getdata(files, ext=0)/4
    flat_data = image_data.flatten()
    summ = a + flat_data
    summpow = b + (flat_data*flat_data)
    
    a = summ
    b = summpow
    i = i+1
    
print(i)    

rumor_mean = summ / i
varianza = (summpow / i) - (rumor_mean*rumor_mean)
rumor_std = np.sqrt(varianza) / math.sqrt(i)

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

plt.yscale("log")
plt.figure()
plt.imshow(a, cmap='plasma')
plt.colorbar()
plt.show()


write_fitsImage(a, "meanpedlongr.fits")
write_fitsImage(b, "stdpedlongr.fits")

