import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al



common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/10mm/1s_G120/'
files_img=['imageCUL_pixCut10.0sigma_CLUcut_5.0sigma.fits']
leg_names=['G=120']


n=0
for i in range(0,len(files_img)):

    fig=plt.figure(i)
    image_data = al.read_image(common_path+files_img[i])
    plt.imshow(image_data)
    plt.title(leg_names[i])
    plt.colorbar()
      
    
plt.show()

