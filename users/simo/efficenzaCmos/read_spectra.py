import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

#import sys
#sys.path.insert(0, '../libs')
import utils_v2 as al
from read_sdd import  pharse_mca
import fit_histogram as fitSimo
import air_attenuation
from  histogramSimo import histogramSimo


mpl.rcParams["font.size"] = 15


    
    


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


def correct_spectrumConst(counts,rebin):

    kcorr=rebin*(0.04/300.)
   # kcorr=1*(0.04/300.)
    
    # correzione con 4% fisso!!!
    counts2=counts
    nbins=len(counts2)
    for i in range(5,nbins+1):
        k=nbins-i
        #print(k," count=",counts[k])
        corr_k=counts2[k]*kcorr
        #print("corr_k=",corr_k)
        # loop su tutti i bin precedenti:
        subtracted=0
        #for j in range(0,k):
        for j in range(0,k):
           
            #print("j=",j)
            subtracted=subtracted+corr_k
            counts2[j]=  counts2[j]-corr_k
            if (counts2[j]<0): counts2[j]=0
            
            
        counts2[k]= counts2[k]+subtracted   

        
        
    return counts2   

def dynamic_constfactor(rebin):
    from scipy import interpolate
    rb=[100,50,25,10,5,1]
   #cf=[0.01,0.0025,0.001,0.0002,0.00005,0.0000015]
    cf=[0.01,0.0025,0.001,0.0002,0.00005,0.000001]

    
    f=interpolate.interp1d(rb,cf, kind='cubic' )
    return (f(rebin))

def correct_spectrumConst2(counts,rebin):

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

def read_allSdd(common_path, mca_file):
    calP0=-0.03544731540487446
    calP1=0.0015013787118821926

    
   
    livetime=[]
    counts=[]
    counts_all=0.
    livetime_all=0
    bins=0
    
    for i in range (0,len(mca_file)):

       p=histogramSimo()
       filename=common_path+mca_file[i]            
       p.read_from_file(filename, 'sdd' )
       print("livetime=",p.sdd_liveTime,"counts=", p.sdd_fastCounts, "RATE=",p.sdd_fastCounts/p.sdd_liveTime,' Hz' )
       print("deadTime=",p.sdd_deadTime)
       
       # calibrazione energia
       p.bins=p.bins*calP1+calP0

       
       mylabel=mca_file[i][0:-4]
       plt.hist(p.bins[:-1],bins=p.bins ,weights=p.counts/p.sdd_liveTime, histtype='step', label=mylabel)
       livetime.append(p.sdd_liveTime)
       counts.append(p.counts)
    
       
       #sum livetimes nad counts:
       if i==0:
           counts_all=p.counts
       else:
            counts_all+=p.counts
       livetime_all+=p.sdd_liveTime
       bins=p.bins

        
    #end loop
    return bins,counts_all,livetime_all


def read_allCMOS(common_path,cmos_eventsFiles,binsSdd,ax2):
   # ax2, to draw individual spectra

    XBINS=2822
    YBINS=4144  
    REBINXY=2.
     
    fig5=plt.figure(5)
    ax=fig5.subplots(1,3)

    
    xbins2d=int(XBINS/REBINXY)
    ybins2d=int(YBINS/REBINXY)
    # retta calibrazione cmos
    calP0=-0.003201340833319255
    calP1=0.003213272145961988
 
    counts=[]
    counts_all=0.
    livetime_all=0
    bins=0
 
    for i in range(0,len(  cmos_eventsFiles)):

        f=common_path+'/cmos/04_07_24/'+cmos_eventsFiles[i]
        print("reading: ",f)
       # w, x,y,size=al.retrive_vectors2(f)
        w, x,y=al.retrive_vectors(f)
        
        print("len w =",w)
        energies=w*calP1+calP0
        #taglio spaziale!!!!
        xc=2064
        yc=2207
        r=502
        #r=5000
        livetime=300.
        mask=np.where( (x-xc)**2+(y-yc)**2<r**2 )
        countsClu, binsE = np.histogram( energies[mask]  , bins =len(binsSdd)-1, range = (binsSdd[0],binsSdd[-1]) )
        
       
        ax2.hist(binsE[:-1],bins=binsE ,weights=countsClu/livetime, histtype='step', label="cmos")
        #plt.legend()

        # plot xy:
        #plt.figure(i+10)
        counts2dClu,  xedges, yedges= np.histogram2d(x[mask],y[mask],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
        counts2dClu=   counts2dClu.T
        #ax=fig5.subplots(2,2)
        im=ax.flatten()[i].imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])


        
        if i==0:
           counts_all=countsClu
        else:
            counts_all+=countsClu 
        livetime_all+=livetime
    plt.title('cmos')

    return binsE,counts_all,livetime_all 

    

if __name__ == "__main__":

    common_path='/home/maldera/IXPE/XCF/data/CMOS_efficiency/'
    #dati sdd"
   
    #mca_file=['10kV_0.018mA_UnPol_0.mca','10kV_0.018mA_UnPol_1.mca','10kV_0.018mA_UnPol_2.mca','10kV_0.018mA_UnPol_3.mca','10kV_0.018mA_UnPol_4.mca']
    mca_file=['10kV_0.018mA_UnPol_1.mca','10kV_0.018mA_UnPol_3.mca','10kV_0.018mA_UnPol_4.mca']

    plt.figure(1)
    binsSdd,counts_all,livetime_all=read_allSdd(common_path+'sdd/SDD_04_07/', mca_file)
               
    #plot SDD
    plt.xlabel('keV')
    plt.ylabel('counts/s [Hz]')
    plt.title('SDD')
    plt.hist(binsSdd[:-1],bins=binsSdd ,weights=counts_all/livetime_all, histtype='step', label="spettro somma")
    plt.legend()  
    plt.title('SDD')

    ### read CMOS data:
    list_name='events_list_pixCut10.0sigma5_CLUcut_10.0sigma.npz'
    cmos_eventsFiles=['/data1/'+list_name, '/data3/'+list_name, 'data4/'+list_name]
  
    fig2=plt.figure(2)
    ax2 = fig2.subplots()
    binsCmos,counts_allCmos,livetimeCmos= read_allCMOS(  common_path, cmos_eventsFiles,binsSdd,ax2)
    ax2.hist(binsCmos[:-1],bins=binsCmos ,weights=counts_allCmos/livetimeCmos, histtype='step', label="spettro somma")
    ax2.legend()  


    print("len(binsSdd=",len(binsSdd))
    print("LEN binsCmos=",len(binsCmos))
  

    #rebinno gli istogrammi
    psdd=histogramSimo()
    psdd.counts=counts_all
    psdd.bins=binsSdd

        
    pcmos=histogramSimo()
    pcmos.counts=counts_allCmos
    pcmos.bins=binsCmos

   
    
    #pcmos.rebin(100)
    #psdd.rebin(100)
    pippo=  pcmos.counts.copy() #!!!!!!
    binCenters=pcmos.bins[0:-1]+(pcmos.bins[1]-pcmos.bins[0])/2.
    corr_counts=correct_spectrumConst2(pippo,1) # nuovo rebin!!!!!! 
    #corr_counts=pippo

    pcmos.rebin(100)
    psdd.rebin(100)
    
    
    print("len dopoRebin (binsSdd=",len(psdd.bins))
    print("LEN dopo rebin  binsCmos=",len(pcmos.bins))
  
    #corr_counts=correct_spectrumLin(pippo,binCenters)

    
    #pcmosCorr=histogramSimo()
    #pcmosCorr.counts=corr_counts
    #pcmosCorr.bins=  pcmos.bins
    
    #pcmos.rebin(1)
    #psdd.rebin(1)
    #pcmosCorr.rebin(1)

    x,y,err= compute_HistRatios(pcmos,psdd)
    #xCorr,yCorr,errCorr= compute_HistRatios(pcmosCorr,psdd)
   
    plt.figure(4)

    print("livetime sdd=",livetime_all)
    print("livetime cmos=",livetimeCmos)
   
    
    ratioLivetime=livetime_all/livetimeCmos
    plt.errorbar(x,y*ratioLivetime,yerr=err*ratioLivetime,fmt='or')
    #plt.errorbar(xCorr,yCorr*ratioLivetime,yerr=errCorr*ratioLivetime,fmt='ob')
    plt.xlim(1.8,7)
    plt.ylim(0,1)
    
    plt.grid()
    
    fig=plt.figure(3)
    ax = fig.subplots()
    pcmos.counts=  pcmos.counts/livetimeCmos
    pcmos.plot(ax,"cmos")

    #pcmosCorr.counts=  pcmosCorr.counts/livetimeCmos
    #pcmosCorr.plot(ax,"cmosCorrected")

    
    psdd.counts=  psdd.counts/livetime_all
    psdd.plot(ax,"sdd")
    plt.legend()
    plt.show()


    


