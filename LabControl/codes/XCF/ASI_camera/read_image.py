
from astropy.io import fits as pf
from matplotlib import pyplot as plt
import numpy as np


#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-09_13_41_13Z/2021-12-09-1341_2-CapObj_0000.FIT'
#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-13_11_52_30Z/2021-12-13-1152_5-CapObj_0000.FIT'
#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-13_13_16_21Z/2021-12-13-1316_3-CapObj_0000.FIT'   # tutto saturo, 16 bit

#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-15_14_36_55Z/2021-12-15-1436_9-CapObj_0000.FIT'   # buio 40 us, 16 bit
#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-15_15_03_24Z/2021-12-15-1503_4-CapObj_0000.FIT' # buio 1s
#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-15_15_19_11Z/2021-12-15-1519_1-CapObj_0000.FIT' # buio 20s
#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-15_15_46_50Z/2021-12-15-1546_8-CapObj_0000.FIT' # buio, 40us, 150gain

#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-20_12_04_51Z/2021-12-20-1204_8-CapObj_0000.FIT' # buio, 40us, 0gain 50wB
#ile_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-20_13_13_13Z/2021-12-20-1313_2-CapObj_0000.FIT' # buio, 40us, 0 gain, 50,50wb, 80 offset 
#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-20_12_19_03Z/2021-12-20-1219_0-CapObj_0000.FIT' # buio, 40us, 120gain, 50,50wb, 80 offset 
file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/mcPherson_verticale/Pd/100ms_G120/2022-12-02-1609_0-CapObj_0047.FIT'



data_f = pf.open(file_path, memmap=True)
data_f.info()

image_data = pf.getdata(file_path, ext=0)/4.
print(image_data.shape)
flat_image=image_data.flatten()


fig, ax = plt.subplots()
ax.hist(flat_image, bins=int(65536/4), range=(0,65536/4)   , alpha=1, histtype='step')
mean = flat_image.mean()
rms = flat_image.std()
s = 'mean=' + str(round(mean, 3)) + "\n"+"RMS=" + str(round(rms,3))
ax.text(0.7, 0.9, s,  transform = ax.transAxes,  bbox = dict(alpha = 0.7))


plt.figure()
plt.imshow(image_data, cmap = 'plasma')
plt.colorbar()
plt.show()


