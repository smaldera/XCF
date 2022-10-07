import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
import ROOT



#####################################################



common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/'
#files_histo=['1s_G120/spectrum_all_raw_cut100.npz', '/1s_G120_bg/spectrum_all.npz', '1s_G120/spectrum_allCLU_cut100.npz', '/1s_G120/spectrum_allCLU_cut50.npz',  '/1s_G120/spectrum_allCLU_cut25.npz', '/1s_G120/spectrum_allCLU_cut500.npz', '/1s_G120/spectrum_allCLU_cut5sigma.npz' ]
#leg_names=['w 55Fe','bg', 'Fe, clustering, cut 100', 'Fe, clustering, cut 50',  'Fe, clustering, cut 25',  'Fe, clustering, cut 500', 'Fe, clustering, cut 5sigma'  ]


files_histo=['/1s_G120/spectrum_all_raw.npz','/1s_G120/spectrum_allCLU_cut25.npz', '/1s_G120/spectrum_allCLU_cut5sigma.npz', '/1s_G120/spectrum_allCLU_cut3sigma_xycut.npz' ]
leg_names=['55Fe', 'Fe, clustering, cut 25',  'Fe, clustering, cut 5sigma' ,  'Fe, clustering, cut 5sigma cut XY' ]




fig, ax = plt.subplots()

for i in range(0,len(files_histo)):
    
    data=np.load(common_path+files_histo[i])
    counts=data['counts']
    bins=data['bins']
    #print ("len(coutsAll)=",len(countsAll) )
    histo=ax.hist(bins[:-1],bins=bins,weights=counts, histtype='step', label=leg_names[i])
   


plt.title('exposure=1s, G=480, 500 frames')
plt.xlabel('ADC ch.')
plt.ylabel('counts')
plt.legend()
plt.show()

