
from astropy.io import fits as pf
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../libs')

import utils as al
from matplotlib import pyplot as plt

def allplots(imagefile, pedfile='none' ):

    image_data=al.read_image(imagefile)/4.

    if pedfile != 'none':

        mean_ped=al.read_image(pedfile)
        # ped subtraction
        image_data=image_data-mean_ped

   # al.isto_all(image_data)   
   # al.plot_image(image_data)

    mask_zeroSupp=np.where(image_data>100)
    zsupp_image=image_data[mask_zeroSupp]
    coords=np.transpose(mask_zeroSupp)
    #al.write_fitsImage(zsupp_image,'prova.fits')
    #plt.imshow(zsupp_image, cmap='plasma')
    print ( zsupp_image)
    print ( coords)
    #print ( zsupp_imag)

    #zero_img=np.zeros((np.shape(image_data)[0],np.shape(image_data)[1]))
    #print(np.shape(zero_img))
    np.savez('pippo',w=zsupp_image,coords=coords)
###################################################################################

###################################################################################

#file_path1='/home/maldera/Desktop/eXTP/ASI294/testImages/testFe/2022-02-11_11_56_05Z_src5sec/2022-02-11-1156_0-CapObj_0000.FIT' 

#file_path2='/home/maldera/Desktop/eXTP/ASI294/testImages/testFe/2022-02-11_11_56_05Z_src5sec/2022-02-11-1156_0-CapObj_0754.FIT' 

file_path_1_noGlass='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_1_noGlass/1000us_G0/2022-05-25-0804_0-CapObj_0006.FIT'


#meanPed_file='mean_pedLong.fits'
#allplots(file_path1,meanPed_file)

data='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/Fe55/source/2022-06-08-1015_1-CapObj_5305.FIT'
meanPed='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/Fe55/bkg/mean_ped.fits'
allplots(data,meanPed)




