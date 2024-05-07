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


mpl.rcParams["font.size"] = 14



def read_histos(path):
    p_list=[]

    for i in range(0,10):
        p=histogramSimo()
        p.read_from_file(path+"spectrum_"+str(i)+".npz", 'npz' )
        p_list.append(p)

    return p_list    

def compute_Ratios(c1,c2,s1,s2):
       r=c1/c2
       rErr=np.sqrt( (s1**2)*((1/c2)**2)+(s2**2)*(c1/(c2**2))**2)
       return r, rErr
       
def compute_HistRatios(p1,p2):
    x=fitSimo.get_centers(p1.bins)
    p1=p1.counts
    p2=p2.counts
    
    s1=np.sqrt(p1)
    s2=np.sqrt(p2)

    y,yerr=compute_Ratios(p1,p2,s1,s2)
        

    return x,y,yerr



if __name__ == "__main__":

    
    pathGPD='/home/maldera/Desktop/eXTP/data/test_finestre/cmos/data_GPD/3x3/'
    pathAir='/home/maldera/Desktop/eXTP/data/test_finestre/cmos/data_100_air/3x3/'
    pathPRC='/home/maldera/Desktop/eXTP/data/test_finestre/cmos/data_100_PRC/3x3/'
    nameSpectrum='spectrumPos_'
    pAir_list=[]
    pGPD_list=[]
    pPRC_list=[]
    ratios=[]
    ratios_err=[]

    ratiosPRC=[]
    ratiosPRC_err=[]
   
    n_bins=3
    nRebin=100
    pAir=histogramSimo()
    pGPD=histogramSimo()
    pPRC=histogramSimo()
    
    p_ratio=histogramSimo()
    p_ratioPRC=histogramSimo()

    fig2, ax = plt.subplots()
    fig2.subplots_adjust(right=0.77)
    for i in range(0,n_bins):
        for j in range(0,n_bins):
            pAir.read_from_file(pathAir+nameSpectrum+str(i)+"_"+str(j)+".npz", 'npz' )
            pAir.rebin(nRebin)
            pAir_list.append(pAir)

            pGPD.read_from_file(pathGPD+nameSpectrum+str(i)+"_"+str(j)+".npz", 'npz' )
            pGPD.rebin(nRebin)
            pGPD_list.append(pGPD)

            pPRC.read_from_file(pathPRC+nameSpectrum+str(i)+"_"+str(j)+".npz", 'npz' )
            pPRC.rebin(nRebin)
            pPRC_list.append(pPRC)



            

            x,y,yerr=compute_HistRatios(pGPD,pAir)
            ax.errorbar(x,y,yerr=yerr, fmt='p-',label='GPD-'+str(i)+'-'+str(j))
            ratios.append(y)
            ratios_err.append(yerr)

            xPRC,yPRC,yerrPRC=compute_HistRatios(pPRC,pAir)
            ax.errorbar(xPRC,yPRC,yerr=yerrPRC, fmt='p-',label='PRC-'+str(i)+'-'+str(j))
            ratiosPRC.append(yPRC)
            ratiosPRC_err.append(yerrPRC)
            
    plt.legend(bbox_to_anchor=(1.01, 1),loc="upper left")     
    ax.set_xlim([1,6])
    ax.set_ylim([0.6,1])
    ax.set_xlabel('E[KeV]')
    ax.set_title('Be trasparency w.r.t air')

    fig3, ax3 = plt.subplots()
    fig3.subplots_adjust(right=0.77)
    for k in range(0,len(ratios)):
        r,rErr= compute_Ratios(ratios[k],ratios[0],ratios_err[k],ratios_err[0])
        ax3.errorbar(fitSimo.get_centers(pGPD.bins),r,yerr=rErr,fmt="-p",label='GPD_'+str(k))
    plt.legend(bbox_to_anchor=(1.01, 1),loc="upper left")    
    ax3.set_xlim([1,6])
    ax3.set_ylim([0.97,1.03])
    ax3.set_xlabel('E[KeV]')
    ax3.grid()
    ax3.set_title('( GPD[i]/Air[i]) / (GPD[0]/Air[0] )')




    fig4, ax4 = plt.subplots()
    fig4.subplots_adjust(right=0.77)
    #fig4.tight_layout()
    for k in range(0,len(ratiosPRC)):
        r,rErr= compute_Ratios(ratiosPRC[k],ratiosPRC[0],ratiosPRC_err[k],ratiosPRC_err[0])
        ax4.errorbar(fitSimo.get_centers(pGPD.bins),r,yerr=rErr,fmt="-p",label='PRC_'+str(k))
        
        #ax4.plot(fitSimo.get_centers(pPRC.bins),ratiosPRC[k]/ratiosPRC[0],"-p",label='PRC_'+str(k))
    plt.legend(bbox_to_anchor=(1.01, 1),loc="upper left")        
    ax4.set_xlim([1,6])
    ax4.set_ylim([0.97,1.03])
    ax4.set_xlabel('E[KeV]')
    ax4.grid()
    ax4.set_title('( PRC[i]/Air[i]) / (PRC[0]/Air[0] )')
    plt.show()


   
 
