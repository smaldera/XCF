import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy
from scipy import interpolate
import sys
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
import air_attenuation
from  histogramSimo import histogramSimo


mpl.rcParams["font.size"] = 14



def linFunc(x,p0,p1):

    return x*p1+p0


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




def dynamic_constfactor(rebin):

    rb=[100,50,25,10,5,1]
    #cf=[0.01,0.0025,0.001,0.0002,0.00005,0.0000015]
    cf=[0.01,0.0025,0.001,0.0002,0.00005,0.000001]

    
    f=interpolate.interp1d(rb,cf, kind='cubic' )
    return (f(rebin))

def correct_spectrumConst(counts,rebin):

    mycf=dynamic_constfactor(rebin)
       
    #kcorr=rebin*(0.04/300.)
    # correzione con 4% fisso!!!
    kcorr=(mycf/rebin)*100.  #(0.01 con rebin 100!) # con rebin100

    print("rebin=",rebin," myCF=",mycf," kcorr=",kcorr )
    
    nbins=len(counts)
    for i in range(1,nbins+1):
        k=nbins-i
       # print(k," count=",counts[k])
        corr_k=counts[k]*kcorr
       # print("corr_k=",corr_k)
        # loop su tutti i bin precedenti:
        subtracted=0
        for j in range(k-20*int((100./rebin)),k):
            #print("j=",j)
            subtracted=subtracted+corr_k
            counts[j]=  counts[j]-corr_k
            if (counts[j]<0): counts[j]=0
            
            
        counts[k]=counts[k]+ subtracted   
    
    return counts   


def correct_spectrumLin(counts,binCenters):


    P1=0.0065
    ycorr=0.05

    #P1=0.0044
    #ycorr=0.03

    nbins=len(counts)
    for i in range(2,nbins+1):
        k=nbins-i
        #print(k," count=",counts[k])
        q=ycorr-P1*binCenters[k+1]
        
        # loop su tutti i bin precedenti:
        subtracted=0
        for j in range(0,k):
            #print("j=",j)
            kcorr=P1*binCenters[j]+q
            if kcorr<0:kcorr=0
            corr_k=counts[k]*kcorr
            subtracted=subtracted+corr_k
            counts[j]=  counts[j]-corr_k
            if (counts[j]<0): counts[j]=0
            
            
        counts[k]+= subtracted   
    
    return counts   



if __name__ == "__main__":

   # path='/home/maldera/IXPE/XCF/data/ASI_55Fe/FF/22April2024/data_3/test_spectrum.npz'
    path='/home/maldera/IXPE/XCF/data/MXR_24ott2025/test3/spectrum_all_eps1.5_pixCut10sigma_CLUcut_10sigma.npz'
    nRebin=1
    p=histogramSimo()
    p.read_from_file(path, 'npz' )
    p.rebin(nRebin)

   # p.counts=p.counts/np.max(p.counts)

    bin_centers=p.bins[0:-1]+(p.bins[1]-p.bins[0])/2. 
    print("bin centers=",bin_centers)
    
    fig=plt.figure(1, (10,10))
    ax1 = fig.subplots()
    p.plot(ax1,'55fe spectrum')

   # plt.show()
    #exit()
    
    #mask=np.where(((bin_centers>0)&(bin_centers<3))|((bin_centers>4)&(bin_centers<5)))

    #ax1.plot(bin_centers[mask],p.counts[mask],'p')
    #popt, pcov=scipy.optimize.curve_fit(linFunc,bin_centers[mask],p.counts[mask])

    #x=np.linspace(bin_centers[mask][0],bin_centers[mask][-1],1000)
    #yfit= linFunc(x,popt[0],popt[1])
    #print("pop1=",popt[1])
    #ax1.plot(x,yfit,'-')


    countsConst= correct_spectrumConst(p.counts.copy(),nRebin)

    #countsLin=correct_spectrumLin(p.counts.copy(), bin_centers)
            
    #pLin=histogramSimo()
    #pLin.counts= countsLin
    #pLin.bins=p.bins

    pConst=histogramSimo()
    pConst.counts= countsConst
    pConst.bins=p.bins
    
    pConst.plot(ax1,'corrected CONST')
    #pLin.plot(ax1,'corrected LIN')
    plt.legend()
        
    plt.show()
