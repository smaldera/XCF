import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../../libs')
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo



# CONFRONTO orizzontale - veritcal Mc_pherson
common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/'
files_histo=['mcPherson_orizz/Ni/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz','mcPherson_orizz/Pd/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz','mcPherson_verticale/Pd/1ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz' ]
leg_names=['Ni orizzontale','Pd orizzontale','Pd verticale']
scale_factor=[100*100*1e-3, 100*100*1e-3   ,100*1e-3]


#files_histo=['mcPherson_orizz/Pd/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz','mcPherson_verticale/Pd/1ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz' ]
#leg_names=['Pd orizzontale norm.','Pd verticale norm.']
#scale_factor=[100*100*1e-3, 100*100*1e-3   ,100*1e-3]


fig, ax = plt.subplots()

popt=[]
for i in range(0,len(files_histo)):


    p=histogramSimo()
    p.read_from_file(common_path+files_histo[i],'npz')
    #p.counts=p.counts/scale_factor[i]
 #   p.normalize(870,900)
    p.plot(ax,leg_names[i])
        
    
plt.title('mcPherson 10kV 0mA - normalized to Pd Lalpha')
plt.xlabel('ADC ch.')
plt.ylabel('counts/s ')
plt.legend()
plt.show()

