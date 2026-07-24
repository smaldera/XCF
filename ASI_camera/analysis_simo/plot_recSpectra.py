import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
#sys.path.insert(0, '../../libs')
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo








#common_path='/home/maldera/Desktop/eXTP/data/CMOS_verticale/test_noise/'

common_path='/home/maldera/IXPE/XCF/data/testReadNoise2/1ms_G120/0/'
#files_histo=['spectrum_all_raw_pixCut7sigma.npz',' spectrum_all_ZeroSupp_pixCut7sigma_CLUcut_20sigma.npz',  'spectrum_all_eps1.5_pixCut7sigma_CLUcut_20sigma.npz']


common_path='/home/maldera/IXPE/XCF/data/'
files_histo=['/testReadNoise2/100ms_G120/0/spectrum_all_raw_pixCut7sigma.npz', 'CMOS_verticale/test_noise/300ms_G120/spectrum_all_raw_pixCut10.0sigma5_parallel.npz']

#common_path='/home/maldera/IXPE/XCF/data/'
#files_histo=['/testReadNoise2/100ms_G120/0/spectrum_all_raw_pixCut7sigma.npz', '/testReadNoise2/100ms_G120/0/spectrum_all_ZeroSupp_pixCut7sigma_CLUcut_20sigma.npz']
leg_names=['1ms new','1ms old']

#leg_names=['raw','soglia','clustering']

scale_factor=[1, 1,1]

fig, ax = plt.subplots()

#for i in range(0,len(files_histo)):
for i in range(0,2):


    p=histogramSimo()
    p.read_from_file(common_path+files_histo[i],'npz')
    p.counts=p.counts/scale_factor[i]
    p.plot(ax,leg_names[i])
        
    
plt.title('test CMOS')
plt.xlabel('ADC ch.')
plt.ylabel('counts')
plt.legend()
plt.show()

