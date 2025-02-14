import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo








common_path='/home/maldera/test/'
files_histo=['data3/spectrum_all_eps1.5_pixCut10sigma_CLUcut_10sigma.npz','data3/test_spectrum.npz', 'data4/spectrum_all_eps1.5_pixCut10sigma_CLUcut_10sigma.npz','data4/test_spectrum.npz']

leg_names=['eventList3','daq3','eventList4','daq4']
scale_factor=[1, 1,1,1]

fig, ax = plt.subplots()

for i in range(0,len(files_histo)):
#for i in range(1,2):


    p=histogramSimo()
    p.read_from_file(common_path+files_histo[i],'npz')
    p.counts=p.counts/scale_factor[i]
    p.plot(ax,leg_names[i])
        
    
plt.title('test CMOS')
plt.xlabel('ADC ch.')
plt.ylabel('counts')
plt.legend()
plt.show()

