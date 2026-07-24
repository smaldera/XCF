from astropy.io import fits as pf
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../libs')

import utils_v2 as al
#import ROOT
#rootObjects=[]




# sensore 2 G120 1s
base_path='/home/maldera/IXPE/XCF/data/testReadNoise2/'

#file_mean=['/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/sensorPXR/G120_10ms_bg/mean_ped.fits', '/home/maldera/Desktop/eXTP/data/misureCMOS_24Jan2023/Mo/G120_10ms_bg/mean_ped.fits']
file_std=['1ms_G120/1/bkg/std_ped.fits', '100ms_G120/1/bkg/std_ped.fits']
file_mean=['1ms_G120/1/bkg/mean_ped.fits', '100ms_G120/1/bkg/mean_ped.fits']
leg_names=['1ms_G120','100ms_G120']



fig=plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0.14, right=0.97, top=0.9, bottom=0.09,hspace=0.250)
ax1=plt.subplot(111)
ax1.set_title('std')

fig2=plt.figure(figsize=(10,7))
fig2.subplots_adjust(left=0.14, right=0.97, top=0.9, bottom=0.09,hspace=0.250)
ax2=plt.subplot(111)
ax2.set_title('mean')




s_std=''
s_mean=''

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
    

    ########
    mean= al.read_image(base_path+file_mean[i])
    counts_mean, bins_mean = np.histogram(mean.flatten(),  bins = 16000, range = (0,16000) ) 
    ax2.hist(bins_mean[:-1], bins = bins_mean, weights = counts_mean, histtype = 'step',label="pedestal mean - "+leg_names[i])

    mean2=mean.flatten().mean()
    rms2=mean.flatten().std()
    s_mean=s_mean+leg_names[i]+":  mean= "+str(round(mean2,3))+" RMS="+str(round(rms2,3))+'\n'
    


ax1.text(0.60, 0.75, s_std,  transform=ax1.transAxes,  bbox=dict(alpha=0.7))
ax1.set_xlim(0,100)
ax1.set_xlabel('adc ch.')


ax2.text(0.60, 0.75, s_mean,  transform=ax2.transAxes,  bbox=dict(alpha=0.7))
ax2.set_xlim(0,16000)
ax2.set_xlabel('adc ch.')


plt.legend()

plt.show()


# wait for stop:   
#input('press any key to continue...')
     
