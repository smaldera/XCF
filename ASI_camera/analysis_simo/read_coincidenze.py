import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo
from cutHotPixels import hotPixels
from tqdm import  tqdm

import fit_histogram as fh
#from landaupy import landau

# small scripts to plot CMOS data from the event list whit cuts
# draws: 2D map,  energy, x-y projections
#


##############3
def fit_landauHistohram(counts,binsE,xmin=-0.5,xmax=2.5, sigma0=0.4):

   mask=(binsE>xmin)&(binsE<xmax)
   mask=mask[:-1] # i bin hanno un elemento in piu'!!!
   max_val=max(counts*mask)
   max_index=np.argmax(counts*mask)
   max_x=binsE[ max_index]+(binsE[1]-binsE[0])/2.
   
   print( "max_val=", max_val)
   print( "max_x=", max_x, " max_index =", max_index)
   
 
   initial_pars=[max_x,sigma0,max_val]
   lims_low=[max_x-0.1,sigma0-0.38,max_val-20]
   lims_up=[max_x+0.1,sigma0+0.2,max_val+20]  
   coeff,pcov,chi2,chi2red= fh.fit_Landau_histogram2(counts,binsE,xmin=xmin,xmax=xmax,  initial_pars= initial_pars, parsBoundsLow=lims_low, parsBoundsUp=lims_up  )
   print ("coeff=",coeff)
   print ("pcov=",pcov)
  
   return  coeff,pcov,chi2,chi2red


######3
def fit_langaussHistohram(counts,binsE,xmin=-0.5,xmax=2.5, sigma0=0.4,gsigma0=0.4):

   mask=(binsE>xmin)&(binsE<xmax)
   mask=mask[:-1] # i bin hanno un elemento in piu'!!!
   max_val=max(counts*mask)
   max_index=np.argmax(counts*mask)
   max_x=binsE[ max_index]+(binsE[1]-binsE[0])/2.
   
   print( "max_val=", max_val)
   print( "max_x=", max_x, " max_index =", max_index)
   
 
   initial_pars=[max_x,sigma0,gsigma0,max_val]
   lims_low=[max_x-0.1,sigma0-0.39,gsigma0-0.39,max_val-20]
   lims_up=[max_x+0.1,sigma0+0.2,gsigma0+0.2,max_val+20]  
   coeff,pcov,chi2,chi2red= fh.fit_Langauss_histogram2(counts,binsE,xmin=xmin,xmax=xmax,  initial_pars= initial_pars, parsBoundsLow=lims_low, parsBoundsUp=lims_up  )
   print ("coeff=",coeff)
   print ("pcov=",pcov)
  
   return  coeff,pcov,chi2,chi2red




######




import argparse
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

parser.add_argument('-dir','--saveDir', type=str,  help='direxctory where npz files are saved', required=False)
parser.add_argument('-calP0','--calP0', type=float,  help='0 cal parameter', required=False,default=-0.003201340833319255)
parser.add_argument('-calP1','--calP1', type=float,  help='1 cal parameter', required=False,default=0.003213272145961988)
parser.add_argument('-rebinxy','--rebinxy', type=int,  help='x-y rebin', required=False,default=80)
parser.add_argument('-nosave','--nosaveHistos', action='store_false',  help="don't save histograms", required=False)
parser.add_argument('-specName','--specName',type=str ,  help="spectrum file name", required=False,default='test_spectrum.npz')
parser.add_argument('-xprojName','--xprojName',type=str ,  help="x-projection file name", required=False,default='test_xproj.npz')
parser.add_argument('-yprojName','--yprojName',type=str ,  help="y-projection file name", required=False,default='test_yproj.npz')
parser.add_argument('-suffix','--suffix',type=str ,  help="suffix in file names", required=False,default='coincidenza')


FIND_HOTPIXELS=True
CUT_HOT_PIXELS=True
PLOT_MAP=True

args = parser.parse_args()
DIR='/home/maldera/Desktop/eXTP/data/testCMOS_coincidenze/29Jul25/'


files_0='/home/maldera/Desktop/eXTP/data/testCMOS_coincidenze/29Jul25/camera0/file_list.txt'
files_1='/home/maldera/Desktop/eXTP/data/testCMOS_coincidenze/29Jul25/camera1/file_list.txt'
ff_0=open(files_0,'r')

# retta calibrazione cmos
calP0=args.calP0
calP1=args.calP1

NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144

REBINXY=args.rebinxy
SAVE_HISTOGRAMS=args.nosaveHistos
spectrum_file_name=args.specName 
xproj_file_name=args.xprojName  
yproj_file_name=args.yprojName
suffix=args.suffix

print("calP0=",calP0,"  calP1=",calP1)
print("Rebin xy=",REBINXY)
print("Save histograms=",SAVE_HISTOGRAMS)
print("spectrum name=",spectrum_file_name)
print("x projection name=",xproj_file_name)
print("y projection name=",yproj_file_name)
print("suffix=",suffix)


xbins2d=int(XBINS/REBINXY)
ybins2d=int(YBINS/REBINXY)

w_all=np.array([])
x_all=np.array([])
y_all=np.array([])
size_all=np.array([])
time_all=np.array([])

for f in ff_0:
    #print(f)
    
    w, x,y,size, time=al.retrive_vectors3(f[:-1])
    #print("LENs w=",len(w)," x=",len(x)," time=",len(time))
    w_all=np.append(w_all,w)
    x_all=np.append(x_all,x)
    y_all=np.append(y_all,y)
    size_all=np.append(size_all,size)
    time_all=np.append(time_all,time)
 
    
print("len w_all ",len(w_all))
print("len x_all ",len(x_all))
print("len y_all ",len(y_all))
print("len time_all ",len(time_all))

# applico hotPixelscut... 
hotPix=hotPixels(x_all=x_all,y_all=y_all,w_all=w_all,size_all=size_all,time_all=time_all,rebin=10)
hotPix.find_HotPixels(n_sigma=4,low_threshold=60, min_counts=10) # low_treshold in ADC, 
hotPix.applyCuts()
w_all,   x_all,  y_all, size_all,time_all=hotPix.get_cutVectors()

print("len w_all  after hotpix cut:",len(w_all))


w1_all=np.array([])
x1_all=np.array([])
y1_all=np.array([])
size1_all=np.array([])
time1_all=np.array([])


############################3
ff_1=open(files_1,'r')

for f in ff_1:
    #print(f)
    w1, x1,y1,size1, time1=al.retrive_vectors3(f[:-1])  
    w1_all=np.append(w1_all,w1)
    x1_all=np.append(x1_all,x1)
    y1_all=np.append(y1_all,y1)
    size1_all=np.append(size1_all,size1)
    time1_all=np.append(time1_all,time1)
 
    
print("len w1_all ",len(w1_all))
print("len x1_all ",len(x1_all))
print("len y1_all ",len(y1_all))
print("len time1_all ",len(time1_all))

# applico hotPixelscut... 
hotPix=hotPixels(x_all=x1_all,y_all=y1_all,w_all=w1_all,size_all=size1_all,time_all=time1_all,rebin=20)
hotPix.find_HotPixels(n_sigma=3,low_threshold=60, min_counts=10) # low_treshold in ADC, 
hotPix.applyCuts()
w1_all,   x1_all,  y1_all, size1_all,time1_all=hotPix.get_cutVectors()

print("len w1_all  after hotpix cut:",len(w1_all))


#myCut=np.where( (w_all>55)&(x_all>5)&(x_all<2808) )
#myCut1=np.where( (w1_all>55)&(x1_all>5)&(x1_all<2808) )

myCut=np.where(w_all>50 )
myCut1=np.where(w1_all>50)



# applico  cut sulle singole:
w1_all=w1_all[myCut1]
x1_all=x1_all[myCut1]
y1_all=y1_all[myCut1]
size1_all=size1_all[myCut1]
time1_all=time1_all[myCut1]

w_all=w_all[myCut]
x_all=x_all[myCut]
y_all=y_all[myCut]
size_all=size_all[myCut]
time_all=time_all[myCut]
print("len w1_all  after w1_all>50:",len(w1_all))
print("len w_all  after w_all>50:",len(w_all))


w1cc=[]
w0cc=[]

minfit=0.2
maxfit=2


##### cerco coincidenze:

for i in tqdm(range(0,len(w_all))):
    cut_1=np.where( np.abs(time1_all-time_all[i])<0.1 )[0]
    #print("len selected=", len(cut_1) )
    if len(cut_1)==1:
      
       w1cc.append(w1_all[cut_1][0])
       w0cc.append(w_all[i])
      # print ("trovata coinc  len=",len(cut_1))







      
# MAKE PLOT2
      
fig2=plt.figure(figsize=(10,10))

# mappa posizioni camera 0
ax1=plt.subplot(221)
ax1.set_title("CMOS 0")
counts2dClu,  xedges, yedges= np.histogram2d(x_all,y_all,bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu=   counts2dClu.T
im=ax1.imshow(np.log10(counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#im=ax1.imshow((counts2dClu), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
fig2.colorbar(im,ax=ax1, orientation='vertical',label="log10(counts)")

# spettro energia cmos 0
ax2=plt.subplot(222)
ax2.set_title("energy spectrum CMOS 0")
countsClu_all, binsE = np.histogram( w_all, bins = int(2*NBINS/4.), range = (-NBINS,NBINS) )
countsClu1_all, binsE = np.histogram( w1_all, bins = int(2*NBINS/4.), range = (-NBINS,NBINS) )

countsClu, binsE = np.histogram( np.array(w0cc)  , bins = int(2*NBINS/4.), range = (-NBINS,NBINS) )
countsClu1, binsE = np.histogram( np.array(w1cc)  , bins = int(2*NBINS/4.), range = (-NBINS,NBINS) )
binsE=binsE*calP1+calP0

ax2.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="cmos 0 - Coincidences")
ax2.hist(binsE[:-1], bins = binsE, weights = countsClu_all, histtype = 'step',label="cmos 0 - ALL  ")


coeff,pcov,chi2,chi2red=  fit_landauHistohram(countsClu,binsE,xmin=minfit,xmax=maxfit)   
coeffLg,pcovLg,chi2Lg,chi2redLg=  fit_langaussHistohram(countsClu,binsE,xmin=minfit,xmax=maxfit)

x=np.arange(minfit,maxfit,0.01)
plt.plot(x, fh.myLandau(x, *coeff), "-r",label='landau')
#plt.plot(x, fh.myLandauGauss(x, *coeffLg), "-r",label='lanGauss')

ax2.set_xlabel('E[keV]')
ax2.set_xlim([-1,5])
ax2.set_yscale('log')
ax2.legend()



# mappa posizioni camera1

ax3=plt.subplot(223)
ax3.set_title("CMOS 1")
counts2dClu1,  xedges, yedges= np.histogram2d(x1_all,y1_all,bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
counts2dClu1=   counts2dClu1.T
im=ax3.imshow(np.log10(counts2dClu1), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
fig2.colorbar(im,ax=ax3, orientation='vertical',label="log10(counts)")



ax4=plt.subplot(224)
ax4.set_title("energy spectrum CMOS 1")
ax4.hist(binsE[:-1], bins = binsE, weights = countsClu1, histtype = 'step',label="cmos 1 - Coincidences ")
ax4.hist(binsE[:-1], bins = binsE, weights = countsClu1_all, histtype = 'step',label="cmos 1 - ALL")

coeff1,pcov1,chi2_1,chi2red_1=  fit_landauHistohram(countsClu1,binsE,xmin=minfit,xmax=maxfit)
coeffLg1,pcovLg1,chi2Lg1,chi2redLg1=  fit_langaussHistohram(countsClu1,binsE,xmin=minfit,xmax=maxfit)
plt.plot(x, fh.myLandau(x, *coeff1), "-r",label='landau')
#plt.plot(x, fh.myLandauGauss(x, *coeffLg1), "-r",label='lanGauss')



ax4.set_xlabel('E[keV]')
ax4.set_xlim([-1,6])
ax4.set_yscale('log')
ax4.legend()


print("==================================")
print("camera0,===>>>> Landau:MPV= ",coeff[0]," +- ",pcov[0][0]**0.5, "chi2 red=",chi2red)
print("coeff=",coeff)
print("cov=",pcov)
print("camera0,=======>>> LanGauss: MPV= ",coeffLg[0]," +- ",pcovLg[0][0]**0.5, "chi2 red=",chi2redLg)
print("coeffLg=",coeffLg)
print("covLg=",pcovLg)
print("\n")
print("==================================")
print("camera1, ======>>>> Landau: MPV= ",coeff1[0]," +- ",pcov1[0][0]**0.5, "chi2 red=",chi2red_1)
print("coeff1=",coeff1)
print("cov1=",pcov1)
print("camera1, ====>>>> LanGauss: MPV= ",coeffLg1[0]," +- ",pcovLg1[0][0]**0.5, "chi2 red=",chi2redLg1)
print("coeffLg1=",coeffLg1)
print("covLg1=",pcovLg1)




fig2=plt.figure(figsize=(10,10))
plt.hist(binsE[:-1], bins = binsE, weights = countsClu, histtype = 'step',label="cmos 0 - Coincidences")
plt.hist(binsE[:-1], bins = binsE, weights = countsClu_all, histtype = 'step',label="cmos 0 - ALL  ")
x=np.arange(minfit,maxfit,0.01)
plt.plot(x, fh.myLandau(x, *coeff), "-r",label='landau')


plt.set_xlabel('E[keV]')
plt.set_xlim([-1,5])
plt.set_yscale('log')
plt.legend()



if SAVE_HISTOGRAMS==True:
  
    print('... saving energy spectrun  in:', spectrum_file_name  )
    np.savez(DIR +suffix+ spectrum_file_name, counts = countsClu,  bins = binsE)
    np.savez(DIR +suffix+'comos1_'+ spectrum_file_name, counts = countsClu1,  bins = binsE)
    
    #np.savez(DIR +suffix+ xproj_file_name, counts = xprojection,  bins = bins_x)
    #np.savez(DIR +suffix+ yproj_file_name, counts = yprojection,  bins = bins_y)
    fig2.savefig(DIR+suffix+'plots.png')
    
plt.show()
