import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
#sys.path.insert(0, '../../libs')
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo








common_path='/home/maldera/Desktop/eXTP/data/CMOS_verticale/test_noise/'
files_histo=['spectrum_all_raw_pixCut10.0sigma5_parallel.npz','spectrum_all_ZeroSupp_pixCut10.0sigma5_CLUcut_10.0sigma_parallel.npz','spectrum_all_eps1.5_pixCut10.0sigma5_CLUcut_10.0sigma_parallel.npz']

leg_names=['raw','soglia','clustering']
scale_factor=[1, 1,1]

fig, ax = plt.subplots()

#for i in range(0,len(files_histo)):
for i in range(0,1):


    p=histogramSimo()
    p.read_from_file(common_path+files_histo[i],'npz')
    p.counts=p.counts/scale_factor[i]
    p.plot(ax,leg_names[i])
        
    
plt.title('test CMOS')
plt.xlabel('ADC ch.')
plt.ylabel('counts')
plt.legend()
plt.show()

