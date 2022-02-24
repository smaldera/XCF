from astropy.io import fits as pf
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import os


files_path = ("C:\\Users\\Acer\\Downloads\\Uni\\Tesi\\Dati\\solo dati FIT - sorgente")   
background_path = ("C:\\Users\\Acer\\Tesi\\meanped.fits")


a = np.zeros((2822, 4144))

background = pf.open(background_path, memmap=True)
back_data = pf.getdata(background_path, ext=0)
#background.info()


for filename in os.listdir(files_path):
    files = os.path.join(files_path, filename)
    data_f = pf.open(files, memmap = True)   
#    data_f.info()
    
    image_data = pf.getdata(files, ext=0)/4.
#    print(image_data)
    all_data = a + image_data - back_data
    a = all_data

print(all_data)
print(back_data)

flat_data = all_data.flatten()

fig, ax = plt.subplots()
ax.hist(flat_data, bins=int(65536/4), range=(0,65536/4)   , alpha=1, histtype='step')
mean = flat_data.mean()
rms = flat_data.std()
s='mean='+str(round(mean,3))+"\n"+"RMS="+str(round(rms,3))
ax.text(0.7, 0.9, s,  transform=ax.transAxes,  bbox=dict(alpha=0.7))

plt.yscale("log")
plt.figure()
plt.imshow(all_data, cmap='plasma')
plt.colorbar()
plt.show()
