import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al

import fit_histogram as fitSimo
from  histogramSimo import histogramSimo







#common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/1s_G120/'
#files_histo=['1s_G120/spectrum_all_raw_cut100.npz', '/1s_G120_bg/spectrum_all.npz', '1s_G120/spectrum_allCLU_cut100.npz', '/1s_G120/spectrum_allCLU_cut50.npz',  '/1s_G120/spectrum_allCLU_cut25.npz', '/1s_G120/spectrum_allCLU_cut500.npz', '/1s_G120/spectrum_allCLU_cut5sigma.npz' ]
#leg_names=['w 55Fe','bg', 'Fe, clustering, cut 100', 'Fe, clustering, cut 50',  'Fe, clustering, cut 25',  'Fe, clustering, cut 500', 'Fe, clustering, cut 5sigma'  ]


#files_histo=['/1s_G120/spectrum_all_raw.npz','/1s_G120/spectrum_allCLU_cut25.npz', '/1s_G120/spectrum_allCLU_cut5sigma.npz', '/1s_G120/spectrum_allCLU_cut3sigma_xycut.npz' ]
#leg_names=['55Fe', 'Fe, clustering, cut 25',  'Fe, clustering, cut 5sigma' ,  'Fe, clustering, cut 5sigma cut XY' ]

#files_histo=['/1s_G120/spectrum_all_raw.npz',   '/1s_G120/spectrum_allCLU_cut25.npz', '/1s_G120/spectrum_allCLU_cut5sigma.npz', '/1s_G120/spectrum_allCLU_cut5sigma_1pixel.npz' ]
#leg_names=['55Fe', 'Fe, clustering, cut 25',  'Fe, clustering, cut 5sigma' ,  'Fe, clustering, cut 5sigma cut 1pixel' ]

#files_histo=['/1s_G120/spectrum_all_raw.npz',   '/1s_G120/spectrum_all_raw_pixMask7.5.npz', '/1s_G120/spectrum_all_pixMask7.5_CLUcut_3sigma.npz']
#leg_names=['Fe raw', 'Fe pixel mask','Fe pixel mask - clustering 3 sigma cut ' ]

#files_histo=['spectrum_all_raw.npz','spectrum_allTEST_eps1_NOpix_CLUcut_5sigma.npz', 'spectrum_allTEST_eps1.5_NOpix_CLUcut_5sigma.npz',  'spectrum_allTEST_eps2_NOpix_CLUcut_5sigma.npz', 'spectrum_allTEST_eps3_NOpix_CLUcut_5sigma.npz', ]
#leg_names=['Fe raw','5 sigma cut, eps=1', '5 sigma cut, eps=1.5',   '5 sigma cut, eps=2',   '5 sigma cut, eps=3']



# CONFRONTO  CLU_CUT
#common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/10mm/1s_G120/'
#files_histo=['spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_3.0sigma.npz','spectrum_all_eps1.5_pixCut10sigma_CLUcut_25sigma.npz',  'spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_2.0sigma.npz', 'spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz']
#leg_names=['CLUcut_3sigma','CLUcut_25sigma','CLUcut_2sigma','CLUcut_5sigma']



# CONFRONTO  Guadagni
#common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/10mm/'
#files_histo=['1s_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz','1s_G240/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz','1s_G280/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz']
#leg_names=['G=120','G=240','G=280']


# CONFRONTO orizzontale - veritcal Mc_pherson
common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/'
files_histo=['mcPherson_orizz/Ni/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz','mcPherson_orizz/Pd/100ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_40.0sigma.npz','mcPherson_verticale/Pd/1ms_G120/spectrum_all_eps1.5_pixCut10.0sigma_CLUcut_5.0sigma.npz' ]

leg_names=['Ni orizzontale','Pd orizzontale','Pd verticale']
scale_factor=[100*100*1e-3, 100*100*1e-3   ,100*1e-3]

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

