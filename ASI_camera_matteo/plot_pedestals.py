from astropy.io import fits as pf
from matplotlib import pyplot as plt
import glob
import numpy as np
import sys
sys.path.insert(0, '/Users/matteo/Desktop/Università/UniTo/Terzo anno/Tesi/XCF-main/ASI_camera/libs')

import utils as al

fig, gmean = plt.subplots()
fig, gstd = plt.subplots()

for i in range (0, len(file_mean)):
    
    image_SW_mean = al.read_image(file_mean[i])
    image_SW_std = al.read_image(file_std[i])
    
    flat_image_mean = image_SW_mean.flatten()
    gmean.hist(flat_image_mean, bins = int(65536/4), range = (0,65536/4), alpha = 1, histtype = 'step')
    mean = flat_image_mean.mean()
    
    
    flat_image_std = image_SW_std.flatten()
    gstd.hist(flat_image_std, bins = int(65536/4), range = (0,65536/4), alpha = 1, histtype = 'step')
    rms = flat_image_std.std()
    
    
gmean.set_title("Multiple means")
gmean.set_ylabel("y")
gmean.set_xlabel("x")
gmean.set_xlim([110, 275])

gstd.set_title("Multiple stds")
gstd.set_ylabel("y")
gstd.set_xlabel("x")
gstd.set_xlim([-5, 300])

gmean.set_box_aspect(1)
gstd.set_box_aspect(1)
    
    
fig.tight_layout(h_pad=2)
plt.show()
     
