import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
import ROOT



#####################################################


#outHisto_name='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_3/Fe55/source/histo_all.npz'
outHisto_name='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/1s_G120/spectrum_all.npz'

common_path='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/'
files_histo=[common_path+'/1s_G120/spectrum_all.npz', common_path+'/1s_G120_bg/spectrum_all.npz']
leg_names=['w 55Fe','bg']

fig, ax = plt.subplots()

for i in range(0,len(files_histo)):
    
    data=np.load(files_histo[i])
    counts=data['counts']
    bins=data['bins']
    #print ("len(coutsAll)=",len(countsAll) )
    histo=ax.hist(bins[:-1],bins=bins,weights=counts, histtype='step', label=leg_names[i])
   



plt.legend()
plt.show()

