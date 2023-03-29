import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo





<<<<<<< HEAD




# CONFRONTO orizzontale - veritcal Mc_pherson
#common_path='/home/xcf/Desktop/xcf_tubo_camera/verticale/Titanio/10KV_0.1mA_120gain_20ms_1000f/'
#files_histo=['spectrum_all_eps1.5_pixCut10.0sigma5_CLUcut_10.0sigma.npz','spectrum_all_raw_pixCut10.0sigma5.npz']

common_path='/home/xcf/Desktop/ASI_polarizzata/Rodio/'

#files_histo=['13Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_3/spectrum_all_eps1.5_pixCut15.0sigma_CLUcut_15.0sigma.npz',  '13Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-10kdx/spectrum_all_eps1.5_pixCut15.0sigma_CLUcut_15.0sigma.npz','13Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-10ksx/spectrum_all_eps1.5_pixCut15.0sigma_CLUcut_15.0sigma.npz', '13Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-20ksx/spectrum_all_eps1.5_pixCut15.0sigma_CLUcut_15.0sigma.npz',  '15Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-15ksx/spectrum_all_eps1.5_pixCut15.0sigma_CLUcut_15.0sigma.npz']
#leg_names=['asse1 0','asse1 +10k','asse1 -10k','asse1 -20k','asse1 -15k']

files_histo=['15Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-15ksx/spectrum_all_eps1.5_pixCut15.0sigma_CLUcut_15.0sigma.npz','15Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-15ksx_asse2_10kdx/spectrum_all_eps1.5_pixCut15.0sigma_CLUcut_15.0sigma.npz' , '15Febb_10KV_0.3mA_120gain_200ms_1000f_h12.99_xtal_asse1-15ksx_asse2_20kdx/spectrum_all_eps1.5_pixCut15.0sigma_CLUcut_15.0sigma.npz'  ]



leg_names=['asse1 -15k','asse1 -15k asse2 +10k','asse1 -15k asse2 +20k']
scale_factor=[1, 1,1]
=======
# CONFRONTO orizzontale - veritcal Mc_pherson
#common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/'
#files_histo=['mcPherson_orizz/Ni/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz','mcPherson_orizz/Pd/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz','mcPherson_verticale/Pd/1ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz' ]

common_path='/home/maldera/Desktop/eXTP/data/xcf_tubo_camera/'
files_histo=['Molibdeno/10KV_0.1mA_120gain_10ms_1000f/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_10.0sigma_CluSize1.npz']
leg_names=['Fe 0.1mA 100ms']
scale_factor=[1000*10*1e-3] # divido per tempo di totale di acquiszione
>>>>>>> 0c99102693bca9d635a90f970ed2f9365c68f146

fig, ax = plt.subplots()

popt=[]
for i in range(0,len(files_histo)):


    p=histogramSimo()
    p.read_from_file(common_path+files_histo[i],'npz')
    p.counts=p.counts/scale_factor[i]
    p.plot(ax,leg_names[i])
        
    
plt.title('Pd Ge111  mcPherson 10kV 0.3mA')
plt.xlabel('ADC ch.')
plt.ylabel('counts')
plt.legend()
plt.show()

