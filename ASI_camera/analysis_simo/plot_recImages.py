import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al



common_path='/home/xcf/Desktop/ASI_polarizzata/Rodio/8Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_ruotato2/'
files_img=['imageCUL_pixCut10.0sigma_CLUcut_10.0sigma.fits']
leg_names=['G=120']


n=0
for i in range(0,len(files_img)):

    fig=plt.figure(i)
    image_data = al.read_image(common_path+files_img[i])
    plt.imshow(image_data)
    plt.title(leg_names[i])
    plt.colorbar()
      
    
plt.show()

