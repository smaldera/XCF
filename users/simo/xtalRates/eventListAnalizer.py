import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import sys
#sys.path.insert(0, '../../libs')
import utils_v2 as al
import fit_histogram as fitSimo
from  histogramSimo import histogramSimo

from cutHotPixels import hotPixels



# small scripts to plot CMOS data from the event list whit cuts
# draws: 2D map,  energy, x-y projections
#


def str2bool(value):
    if value.lower() in {'true', 't', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'f', 'no', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")





#import argparse
#formatter = argparse.ArgumentDefaultsHelpFormatter
#parser = argparse.ArgumentParser(formatter_class=formatter)
#parser.add_argument('-in','--inFile', type=str,  help='txt file with list of npz files', required=True)
#parser.add_argument('-dir','--saveDir', type=str,  help='direxctory where npz files are saved', required=False)
#parser.add_argument('-calP0','--calP0', type=float,  help='0 cal parameter', required=False,default=-0.003201340833319255)
#parser.add_argument('-calP1','--calP1', type=float,  help='1 cal parameter', required=False,default=0.003213272145961988)
#parser.add_argument('-rebinxy','--rebinxy', type=int,  help='x-y rebin', required=False,default=20)
#parser.add_argument('-nosave','--nosaveHistos', action='store_false',  help="don't save histograms", required=False)
#parser.add_argument('-specName','--specName',type=str ,  help="spectrum file name", required=False,default='test_spectrum.npz')
#parser.add_argument('-xprojName','--xprojName',type=str ,  help="x-projection file name", required=False,default='test_xproj.npz')
#parser.add_argument('-yprojName','--yprojName',type=str ,  help="y-projection file name", required=False,default='test_yproj.npz')
#parser.add_argument('-suffix','--suffix',type=str ,  help="suffix in file names", required=False,default='')
#parser.add_argument('--hotPixelsCut',type=str2bool,  help="if true enable hot pixel cut", required=False,default=True)



NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144

class readEventList:
    
    def __init__(self, fileList,calP0=-0.003201340833319255, calP1=0.003213272145961988, hotCut=False, outpath='.',suffix='',rebin=20 ):
    

        self.FIND_HOTPIXELS=True
        self.CUT_HOT_PIXELS=True
        self.PLOT_MAP=True

        self.infile=fileList
        #DIR = args.saveDir
        #ff=open(args.inFile,'r')

        
        # retta calibrazione cmos
        self.calP0=calP0
        self.calP1=calP1


        self.REBINXY=rebin
        self.SAVE_HISTOGRAMS=True
        self.spectrum_file_name='test_spectrum.npz' 
        self.xproj_file_name= 'test_xproj.npz'
        self.yproj_file_name='test_yproj.npz'
        self.suffix=suffix
        self.path=outpath
        self.hotCut=hotCut

        print("calP0=",calP0,"  calP1=",calP1)
        # print("Rebin xy=",REBINXY)
        # print("Save histograms=",SAVE_HISTOGRAMS)
        # print("spectrum name=",spectrum_file_name)
        # print("x projection name=",xproj_file_name)
        # print("y projection name=",yproj_file_name)
        # print("suffix=",suffix)
        # print("Pixel Cut=",args.hotPixelsCut, "FIND_HOTPIXELS= ",FIND_HOTPIXELS," CUT_HOT_PIXELS=",CUT_HOT_PIXELS)
        

        self.xbins2d=int(XBINS/self.REBINXY)
        self.ybins2d=int(YBINS/self.REBINXY)


    def readFiles(self): 

       ff= open(self.infile,'r')
        
       self.w_all=np.array([])
       self.x_all=np.array([])
       self.y_all=np.array([])
       self.size_all=np.array([])
       self.timestamp_all=np.array([])

       for f in ff:
           print(f[:-1])
           #w, x,y=al.retrive_vectors(f[:-1])
           #w, x,y,size=al.retrive_vectors2(f[:-1])
           w, x,y,size,timestamp=al.retrive_vectors3(f[:-1])
           self.w_all=np.append(self.w_all,w)
           self.x_all=np.append(self.x_all,x)
           self.y_all=np.append(self.y_all,y)
           self.size_all=np.append(self.size_all,size)
           self.timestamp_all=np.append(self.timestamp_all,timestamp)

       print("len w_all ",len(self.w_all))
       print("len x_all ",len(self.x_all))
       print("len y_all ",len(self.y_all))


    def HotPixelsCut(self, rebinVal=20):

        hotPix=hotPixels(x_all=self.x_all,y_all=self.y_all,w_all=self.w_all,size_all=self.size_all,rebin=renbinVal)
        hotPix.find_HotPixels(n_sigma=20,low_threshold=30, min_counts=50) # low_treshold in ADC, 
        hotPix.save_cuts(DIR+'/cuts.npz')
             
        #hotPix=hotPixels(x_all=x_all,y_all=y_all,w_all=w_all,size_all=size_all,rebin=REBINXY)
        hotPix.retrive_cuts(DIR+'/cuts.npz')
        hotPix.applyCuts()
        self.w_all,   self.x_all,  self.y_all, self.size_all=hotPix.get_cutVectors() #!!!!!!!!!! tempi??????
                    

    def apply_customMask(self):          
        
        #mask2=np.where(  ~((x_all==1732)&(y_all==4143))  )
        #mask2=np.where(x_all<1e10)

        #energy_all=w_all*calP1+calP0
        #myCut=np.where( (w_all>50))
        #myCut=np.where( (w_all>50)&(energy_all<2.4)&(energy_all>2.2))
        
        #myCut=np.where( ((w_all)>50)&(x_all>5)&(x_all<2808))
        #myCut=np.where( ((size_all)>1))

        #r=((17/3.1415)**0.5)/4.6e-3
        #mask2=np.where( (self.w_all>0)&( (  ((self.x_all-1460)**2+(self.y_all-2000)**2)< r**2 )  ))


        mask2=np.where( self.w_all>40)
        

        self.x_all= self.x_all[mask2]
        self.y_all= self.y_all[mask2] 
        self.w_all= self.w_all[mask2]
        self.size_all=self.size_all[mask2]
        self.timestamp_ll=  self.timestamp_all[mask2]
        print("len w_all dopo cut custm  ",len(self.w_all))
#===============

    def mappa(self, plot=False):

     
        counts2dClu,  self.xedges, self.yedges= np.histogram2d(self.x_all, self.y_all,bins=[self.xbins2d, self.ybins2d ],range=[[0,XBINS],[0,YBINS]])
        self.counts2dClu=   counts2dClu.T

        if plot==True:
          fig1=plt.figure()
          ax1=plt.subplot(111)
          im=ax1.imshow(np.log10(self.counts2dClu), interpolation='nearest', origin='lower',  extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
         # plt.colorbar()
          plt.title('log10(counts)')
        #  ax1.legend()
          plt.show()


    def spettri(self,plot=False):

       #energy_all=self.w_all*calP1+calP0
       self.countsClu, binsE = np.histogram(self.w_all  , bins = int(2*NBINS/2.), range = (-NBINS,NBINS) )
       self.binsE=binsE*self.calP1+self.calP0
       if plot==True:
          fig2=plt.figure()
          ax2=plt.subplot(111)     
          ax2.hist(self.binsE[:-1], bins = self.binsE, weights = self.countsClu, histtype = 'step',label="energy w. clustering")
          ax2.set_xlabel('E[keV]')
          #ax2.set_xlim([0,12])
          ax2.set_yscale('log')
          ax2.legend()
          plt.show()   


    def projections(self,plot=False):

          self.xprojection, self.bins_x = np.histogram( self.x_all  , bins =self.xbins2d , range = (0,XBINS) )
          self.yprojection, self.bins_y = np.histogram( self.y_all  , bins =self.ybins2d , range = (0,YBINS) )

          if plot==True:
              fig3=plt.figure()
              ax3=plt.subplot(211)  
              ax3.hist(self.bins_x[:-1], bins = self.bins_x, weights = self.xprojection, histtype = 'step',label="x-projection")
              ax3.legend()
              ax3.set_yscale('log')

              ax4=plt.subplot(212)  
              ax4.hist(self.bins_y[:-1], bins = self.bins_y, weights = self.yprojection, histtype = 'step',label="y-projection")
              ax4.legend()
              ax4.set_yscale('log')
              plt.show()
              
              

    def saveHistograms(self):
        
        spectrum_file_name='/spectrum'
        xproj_file_name='/xprojection'
        yproj_file_name= '/yprojection'
        
        
        print('... saving energy spectrun  in:',self.path+spectrum_file_name+self.suffix  )
        np.savez(self.path + spectrum_file_name+self.suffix+'.npz', counts = self.countsClu,  bins = self.binsE)
        np.savez(self.path + xproj_file_name+self.suffix+'.npz', counts = self.xprojection,  bins = self.bins_x)
        np.savez(self.path + yproj_file_name+self.suffix+'.npz', counts = self.yprojection,  bins = self.bins_y)
        #fig2.savefig(DIR+suffix+'plots.png')
    

            

    def analyze_all(self,plot_all=True):
       
       #read npx file
       print("reading events files list:",self.infile )
       self. readFiles()

       #cut pixel rumorosi:
       if self.hotCut==True:
           self. HotPixelsCut()
           
       # applica tagli:
       self.apply_customMask()

       # mappa:
       self.mappa(plot=False)

       # spettri
       self.spettri(plot=False)

       # proiezioni:
       self.projections(plot=False)
       # save histos:
       self.saveHistograms()

       
       if plot_all==True:
           fig4=plt.figure(4)

           ax4_1=plt.subplot(221)
           im=ax4_1.imshow(np.log10(self.counts2dClu), interpolation='nearest', origin='lower',  extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])  
           ax4_1.set_xlabel('x pixels')
           ax4_1.set_ylabel('y pixels')
        

           ax4_2=plt.subplot(222)
           ax4_2.hist(self.binsE[:-1], bins = self.binsE, weights = self.countsClu, histtype = 'step',label="energy w. clustering")
           ax4_2.set_xlabel('E[keV]') 
           ax4_2.set_yscale('log')
           ax4_2.set_xlim([0,12])

           ax4_3=plt.subplot(223)
           ax4_3.hist(self.bins_x[:-1], bins = self.bins_x, weights = self.xprojection, histtype = 'step',label="x-projection")
           ax4_3.legend()
           ax4_3.set_yscale('log')

           ax4_4=plt.subplot(224)  
           ax4_4.hist(self.bins_y[:-1], bins = self.bins_y, weights = self.yprojection, histtype = 'step',label="y-projection")
           ax4_4.legend()
           ax4_4.set_yscale('log')
           fig4.savefig(self.path+'/plotsAll'+self.suffix+'.png')
    
           plt.show()

           
           
       

if __name__ == '__main__':

    infile1='/home/maldera/IXPE/XCF/data/testAllimentoXtals/1/eventFileList.txt'

    print("inizializzo la classe")
    myReadEventList=  readEventList(infile1,path='/home/maldera/IXPE/XCF/data/testAllimentoXtals/1/' )
    print("analizzo tutto!!!!!!")
    myReadEventList.analyze_all(plot_all=True)
    
       

