import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from eventListAnalizer import readEventList

import os


def applycut(eva):

    # definisco taglio
    r=((17/3.1415)**0.5)/4.6e-3
    mask2=np.where( (eva.w_all>0)&( (  (eva.x_all-1460)**2+(eva.y_all-2000)**2)< r**2 )  )

    #applico taglio
    eva.x_all= eva.x_all[mask2]
    eva.y_all= eva.y_all[mask2] 
    eva.w_all= eva.w_all[mask2]
    eva.size_all=eva.size_all[mask2]
    eva.timestamp_ll= eva.timestamp_all[mask2]

    #rianalizzo
    eva.mappa(plot=False)
    eva.spettri(plot=False)
    #eva.analyze_all(plot_all=False)

    return eva
    




base_path='/home/maldera/IXPE/XCF/data/testAllimentoXtals/'


paths=['1','2','3','4','5','8']
phisteps=['0','-5k','-10k','-15k','-20k','-35k']
phi=[0,5,10,15,20,35]
rates=[]
counts=[]
exposure=[]

fig1,mapaxes=plt.subplots(2,3)
fig2,map2axes=plt.subplots(2,3)
fig3,Eax=plt.subplots(1,1)




i=0
for pp in paths:
    path=base_path+pp
    cmd='ls '+path+'/events_list*.npz > '+path+'/eventFileList.txt'
    print(cmd)
    os.system(cmd)

    infile=base_path+pp+'/eventFileList.txt'
    eva= readEventList(infile,outpath=base_path+pp,rebin=40)

    
    
    
    print("analizzo tutto!!!!!!")
    eva.analyze_all(plot_all=False)
    im=mapaxes.flat[i].imshow(np.log10(eva.counts2dClu), interpolation='nearest', origin='lower',  extent=[eva.xedges[0], eva.xedges[-1], eva.yedges[0], eva.yedges[-1]])
    mapaxes.flat[i].set_title('phi='+phisteps[i])


    #applicu cut sdd
    eva_cut= applycut(eva)
    #eva_cut=eva 
   
    im=map2axes.flat[i].imshow(np.log10(eva_cut.counts2dClu), interpolation='nearest', origin='lower',  extent=[eva_cut.xedges[0], eva_cut.xedges[-1], eva_cut.yedges[0], eva_cut.yedges[-1]])
    map2axes.flat[i].set_title('phi='+phisteps[i])

    
    #plot spettri
    Eax.hist(eva_cut.binsE[:-1], bins = eva_cut.binsE, weights = eva_cut.countsClu, histtype = 'step',label='phi='+phisteps[i] )
    #Eax.set_title(phisteps[i])
    Eax.set_xlabel('E[keV]') 
    Eax.set_yscale('log')
    Eax.set_xlim([0,12])

    
    NevCut=len(eva_cut.w_all)
    print("nev cut=",NevCut)

        
    # read n.frames:
    n_framesfile=base_path+pp+'/numero_eventi'
    n=-1
    with open( n_framesfile) as f:
        
        line = f.readline() 
        n=int(line)
        print("line=",line," n=",n)

        
    rates.append(NevCut/(n*0.3))
    exposure.append(n*0.3)
    counts.append(NevCut)
    print ("rate ev=",rates[i])
    i=i+1





fig4,ax=plt.subplots(1,1)
#ax.plot(phi,rates,'or--')

err=np.sqrt(np.array(counts))/np.array(exposure)

ax.errorbar(phi,rates,yerr=err,fmt='or--')
ax.set_ylabel('rate in SDD region [Hz]') 
ax.set_xlabel('phi steps (x -1000)')
ax.grid()    
plt.legend()
plt.show()    
