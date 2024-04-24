import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '/home/maldera/Desktop/eXTP/softwareXCF/XCF/libs/')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
import air_attenuation
from  histogramSimo import histogramSimo


mpl.rcParams["font.size"] = 15



def read_histos(path):
    p_list=[]

    for i in range(0,10):
        p=histogramSimo()
        p.read_from_file(path+"spectrum_"+str(i)+".npz", 'npz' )
        p_list.append(p)

    return p_list    

def compute_ratios(p1,p2):
    x=fitSimo.get_centers(p1.bins)
    p1=p1.counts
    p2=p2.counts
    
    s1=np.sqrt(p1)
    s2=np.sqrt(p2)
    
        
    y=p1/p2
    yerr=np.sqrt( (s1**2)*((1/p2)**2)+(s2**2)*(p1/(p2**2))**2)
        

    return x,y,yerr



if __name__ == "__main__":

    
    pathGPD='/home/maldera/Desktop/eXTP/data/test_finestre/cmos/data_GPD/'
    pathAir='/home/maldera/Desktop/eXTP/data/test_finestre/cmos/data_100_air/'
    pAir_list=[]
    pGPD_list=[]
    ratios=[]
    ratios_err=[]
   
    n_bins=3
    pAir=histogramSimo()
    pGPD=histogramSimo()
    p_ratio=histogramSimo()

    fig2, ax = plt.subplots()

    for i in range(0,n_bins):
        for j in range(0,n_bins):
            pAir.read_from_file(pathAir+"spectrumPos_"+str(i)+"_"+str(j)+".npz", 'npz' )  
            pAir_list.append(pAir)
            pGPD.read_from_file(pathGPD+"spectrumPos_"+str(i)+"_"+str(j)+".npz", 'npz' )  
            pGPD_list.append(pGPD)

            x,y,yerr=compute_ratios(pGPD,pAir)
            ax.errorbar(x,y,yerr=yerr, fmt='p-',label=str(i)+'-'+str(j))
            ratios.append(y)
            ratios_err.append(yerr)
    plt.legend()     


    fig3, ax3 = plt.subplots()

    for k in range(0,len(ratios)):
        ax3.plot(fitSimo.get_centers(pGPD.bins),ratios[k]/ratios[0],"-p",label=str(k))
    
    
  

    plt.show()


   
 
