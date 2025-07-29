import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

#import sys
#sys.path.insert(0, '/home/maldera/Desktop/eXTP/softwareXCF/XCF/libs/')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
import air_attenuation
from  histogramSimo import histogramSimo


mpl.rcParams["font.size"] = 15

def compute_Ratios(c1,c2,s1,s2):
       r=c1/c2
       rErr=np.sqrt( (s1**2)*((1/c2)**2)+(s2**2)*(c1/(c2**2))**2)
       return r, rErr


def leggi_fitResults(nomefile):
    n_line=0
    f=open(nomefile)
    norms=[]
    normsErr=[]
    means=[]
    sigmas=[]
    for line in f:
        print(line[:-1].split())
        splitted=line[:-1].split()
        if n_line==0:
            print (splitted[0].split("=")[1])
            print (splitted[1].split("=")[1])
        else:
            print("AAA ",splitted[0].split("=")[1], " ",splitted[1].split("=")[1], " ",splitted[2].split("=")[1] )
            norms.append(float(splitted[0].split("=")[1]) )
            normsErr.append(float(splitted[1].split("=")[1]) )
            means.append(float(splitted[2].split("=")[1]))
            sigmas.append(float(splitted[3].split("=")[1]))
           
        n_line+=1    
    print("norms list= ",norms )                  
    return (np.array(norms),np.array(normsErr), np.array(means),np.array(sigmas))                  

                          
if __name__ == "__main__":

    
    
    common_path='/home/maldera/Desktop/eXTP/data/test_finestre/Norcada/scan/'
    Nrebin=1
    
    for i  in range(1,12):
           
           fileAir=common_path+'air_'+str(i)+'/spectrumPos_all.npz'
           fileWin=common_path+'win_'+str(i)+'/spectrumPos_all.npz'

           pWin=histogramSimo()
           pWin.read_from_file(fileWin, 'npz' )
           pWin.rebin(Nrebin)

           pAir=histogramSimo()
           pAir.read_from_file(fileAir, 'npz' )
           pAir.rebin(Nrebin)

           plt.hist(pWin.bins[:-1],bins=pWin.bins ,weights=pWin.counts, histtype='step', label='Norcada')
           plt.hist(pAir.bins[:-1],bins=pAir.bins ,weights=pAir.counts, histtype='step', label='Air')
            
           plt.xlabel('keV')
           plt.ylabel('counts')
           plt.legend()  
           
           plt.show()




