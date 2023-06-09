import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al



common_path='/home/maldera/Desktop/eXTP/data/ASI_polarizzata/Rodio/8Febb_10KV_0.1mA_120gain_200ms_1000f_h12.99/'
files_img=['imageRaw_pixCut10.0sigma.fits']
leg_names=['G=120']


n=0
for i in range(0,len(files_img)):

    fig=plt.figure(i)
    image_data = al.read_image(common_path+files_img[i])
    plt.imshow(image_data,origin='lower')
    plt.title(leg_names[i])
    plt.colorbar()
      
    
plt.show()

