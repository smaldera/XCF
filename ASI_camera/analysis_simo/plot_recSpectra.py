import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo





# CONFRONTO orizzontale - veritcal Mc_pherson
#common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/'
#files_histo=['mcPherson_orizz/Ni/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz','mcPherson_orizz/Pd/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz','mcPherson_verticale/Pd/1ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz' ]

common_path='/home/maldera/Desktop/eXTP/data/xcf_tubo_camera/'
files_histo=['Molibdeno/10KV_0.1mA_120gain_10ms_1000f/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_10.0sigma_CluSize1.npz']
leg_names=['Fe 0.1mA 100ms']
scale_factor=[1000*10*1e-3] # divido per tempo di totale di acquiszione

fig, ax = plt.subplots()

popt=[]
for i in range(0,len(files_histo)):


    p=histogramSimo()
    p.read_from_file(common_path+files_histo[i],'npz')
    p.counts=p.counts/scale_factor[i]
    p.plot(ax,leg_names[i])
        
    
plt.title('mcPherson 10kV 0mA')
plt.xlabel('ADC ch.')
plt.ylabel('counts/s')
plt.legend()
plt.show()

