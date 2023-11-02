import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al



common_path='/home/maldera/Desktop/eXTP/data/ASI_newSphere/Pd_6KV_0.1ma_200ms_g120_1000f/'
files_img=['imageCUL_pixCut10.0sigma5_CLUcut_10.0sigma_parallel.fits']
leg_names=['Pd_6KV_0.1mA']


n=0
for i in range(0,len(files_img)):

    fig=plt.figure(i)
    image_data = al.read_image(common_path+files_img[i])
    plt.imshow(image_data,origin='lower')
    plt.title(leg_names[i])
    plt.colorbar()
      
    
plt.show()

