from astropy.io import fits as pf
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../libs')

import utils as al
#import ROOT
#rootObjects=[]




# sensore 2 G120 1s
base_path='/home/maldera/Desktop/eXTP/data/CMOS_verticale/test_noise/'

#file_mean=['/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/sensorPXR/G120_10ms_bg/mean_ped.fits', '/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/G120_10ms_bg/mean_ped.fits']
file_std=['1ms_G120/std_ped.fits', '300ms_G120/std_ped.fits']
leg_names=['1ms_G120','300ms_G120']



fig=plt.figure(figsize=(10,7))
fig.subplots_adjust(left=0.14, right=0.97, top=0.9, bottom=0.09,hspace=0.250)
ax1=plt.subplot(111)
ax1.set_title('std')


s_std=''
for i in range (0, len(file_std)):
    #mean= al.read_image(base_path+file_mean[i])
    std= al.read_image(base_path+file_std[i])

     #flat_image = image_data.flatten()

    # spettro "raw"
    counts_rms, bins_rms = np.histogram(std.flatten(),  bins = 1600, range = (0,200) ) 
    ax1.hist(bins_rms[:-1], bins = bins_rms, weights = counts_rms, histtype = 'step',label="pedestal std - "+leg_names[i])

    mean=std.flatten().mean()
    rms=std.flatten().std()
    s_std=s_std+leg_names[i]+":  mean= "+str(round(mean,3))+" RMS="+str(round(rms,3))+'\n'
    
    


ax1.text(0.60, 0.75, s_std,  transform=ax1.transAxes,  bbox=dict(alpha=0.7))
ax1.set_xlim(0,100)
ax1.set_xlabel('adc ch.')
plt.legend()

plt.show()


# wait for stop:   
#input('press any key to continue...')
     
