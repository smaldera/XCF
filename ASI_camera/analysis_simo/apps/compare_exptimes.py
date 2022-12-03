import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo



            
#####################################################


# CONFRONTO  tempi esposizione
common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/2mm/'
files_histo=['1s_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz','10s_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz','01s_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz']
leg_names=['1s','10s','0.1s']



fig, ax = plt.subplots()

popt=[]
for i in range(0,len(files_histo)):


   # plot_spectrum(common_path+files_histo[i],leg_names[i],ax) 
    p=histogramSimo()
    p.get_from_file(common_path+files_histo[i])
    p.normalize( 500,4000 )
    p.plot(ax,leg_names[i])
        
    
plt.title('G=120, 2mm collimator')
plt.xlabel('ADC ch.')
plt.ylabel('counts')
plt.legend()
plt.show()

