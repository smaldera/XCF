import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
from pedestal import bg_map
import clustering_cmos

from multiprocessing import Process,  Queue
from tqdm import tqdm


class analize_v2():
    """
    """
    def __init__(self, fileList,bg_path):
        
        self.create_bg_map = False
        self.NBINS=16384  # n.canali ADC (2^14)
        self.XBINS=2822
        self.YBINS=4144
        self.PIX_CUT_SIGMA=10. # cut per pixel rumorosi
        self.CLU_CUT_SIGMA=10. # clustering cut
        self.REBINXY=20.
        self.APPLY_CLUSTERING=True
        self.SAVE_EVENTLIST=True
        self.myeps=1.5 # clustering DBSCAN
        self.make_rawSpectrum=False
        
        self.xbins2d=int(self.XBINS/self.REBINXY)
        self.ybins2d=int(self.YBINS/self.REBINXY)


        self.pixMask_suffix='_pixCut'+str(self.PIX_CUT_SIGMA)+'sigma5'
        self.cluCut_suffix='_CLUcut_'+str(self.CLU_CUT_SIGMA)+'sigma'
        self.fileList=fileList
        self.bg_path=bg_path

      
    def reset_allVariables(self):
        x = []
        self.countsAll2dClu=np.array(x)
        self.xedges=np.array(x)
        self.yedges=np.array(x)
        self.countsAll=np.array(x)
        self.bins=np.array(x)
        self.countsAllZeroSupp=np.array(x)
        self.countsAllClu=np.array(x)
        self.h_cluSizeAll=np.array(x)
        self.countsAll2dRaw=np.array(x)
        self.xedgesRaw=np.array(x)
        self.yedgesRaw=np.array(x)
        
         
        self.countsAll, self.bins = np.histogram(x, bins = 2*self.NBINS, range = (-self.NBINS,self.NBINS))
        self.countsAllZeroSupp, self.bins = np.histogram(x, bins = 2*self.NBINS, range = (-self.NBINS,self.NBINS))
        self.countsAllClu, self.bins = np.histogram(x,  bins = 2*self.NBINS, range = (-self.NBINS,self.NBINS))
        self.h_cluSizeAll,self.binsSize=np.histogram(x,bins=100, range=(0,100))

        # creo histo2d vuoto:
        self.countsAll2dClu,  self.xedges, self.yedges=       np.histogram2d(x,x,bins=[self.xbins2d, self.ybins2d],range=[[0,self.XBINS],[0,self.YBINS]])
        self.countsAll2dRaw,  self.xedgesRaw, self.yedgesRaw= np.histogram2d(x,x,bins=[self.xbins2d, self.ybins2d],range=[[0,self.XBINS],[0,self.YBINS]])

        # x_all=np.empty(0)
        #y_all=np.empty(0)
        self.x_allClu=np.empty(0)
        self.y_allClu=np.empty(0)
        self.w_all=np.empty(0)

     
        
        
    def do_analyze(self,queue,n_job):
     
        if self.create_bg_map == True:
           bg_map(self.bg_path, self.bg_path + 'mean_ped.fits', self.bg_path + 'std_ped.fits', draw = 0 )

        # inizio analisi...
        # leggo files pedestal (mean e rms)
        pedfile  = self.bg_path + 'mean_ped.fits'
        mean_ped = al.read_image(pedfile)
        pedSigmafile  = self.bg_path + 'std_ped.fits'
        rms_ped = al.read_image(pedSigmafile)

        # MASCHERA PIXEL RUMOROSI
        rms_pedCut=np.mean(rms_ped)+self.PIX_CUT_SIGMA*np.std(rms_ped)
        #mySigmaMask=np.where( (rms_ped>10)&(mean_ped>500) )
        mySigmaMask=np.where( (rms_ped>rms_pedCut) )
        self.reset_allVariables()
    
        
        # inizio loop sui files
        bar_prefix='processing data chunk '+str(n_job)
        for image_file in tqdm(self.fileList, desc=bar_prefix,  colour='green',position=n_job-1):             
            # read image:
            image_data = al.read_image(image_file)/4.
            # subtract pedestal:
            image_data = image_data -  mean_ped #

            #applica maschera
            image_data[mySigmaMask]=0 # maschero tutti i pixel con RMS pedestal > soglia 

            if self.make_rawSpectrum==True:
                flat_image = image_data.flatten()
                counts_i, bins_i = np.histogram(flat_image,  bins = 2*self.NBINS, range = (-self.NBINS,self.NBINS) ) 
                self.countsAll = self.countsAll + counts_i

            #################
            #ZERO SUPPRESSION
            # applico selezione su carica dei pixel
            # supp_coords, supp_weights=al.select_pixels2(image_data, 150)
            supp_coords, supp_weights=al.select_pixels_RMS(image_data, rms_ped, self.CLU_CUT_SIGMA)
     
            if len( supp_weights)==0:
                   print ('no pixel above zero supp. threshold... skipping image')
                   continue
    
            # salvo pixel che sopravvivono alla selezione:
            zeroSupp_trasposta= supp_coords

            
            #istogramma 2d immagine raw dopo zero suppression:
            counts2dRaw,  xedgesRaw, yedgesRaw= np.histogram2d(zeroSupp_trasposta[0],zeroSupp_trasposta[1],bins=[self.xbins2d, self.ybins2d ],range=[[0,self.XBINS],[0,self.YBINS]])
            self.countsAll2dRaw=self.countsAll2dRaw+counts2dRaw

            #spettro dopo zeroSuppression
            countsZeroSupp_i, bins_i = np.histogram( supp_weights,  bins = 2*self.NBINS, range = (-self.NBINS,self.NBINS) ) 
            self.countsAllZeroSupp =self.countsAllZeroSupp  +  countsZeroSupp_i

            #CLUSTERING
            if self.APPLY_CLUSTERING:
       
                   # test clustering.... # uso v3 per avere anche le posizioni
                   w_clusterAll, clu_coordsAll, clu_sizes, clu_baryCoords    =clustering_cmos.clustering_v3(np.transpose(supp_coords),supp_weights,myeps=self.myeps) 
                   cluBary_trasposta= clu_baryCoords.transpose()
   
                   if self.SAVE_EVENTLIST:
                       self.x_allClu=np.append(self.x_allClu,cluBary_trasposta[0])
                       self.y_allClu=np.append(self.y_allClu,cluBary_trasposta[1])
                       self.w_all=np.append(self.w_all,w_clusterAll)
        
                       # istogramma 2d dopo clustering solo baricentri!!!!
                       counts2dClu,  xedges, yedges= np.histogram2d(cluBary_trasposta[0],cluBary_trasposta[1],bins=[self.xbins2d, self.ybins2d ],range=[[0,self.XBINS],[0,self.YBINS]])
                       self.countsAll2dClu=self.countsAll2dClu+ counts2dClu

                       # istogramma spettro dopo il clustering
                      
                       countsClu_i, bins_i = np.histogram(  w_clusterAll, bins = 2*self.NBINS, range = (-self.NBINS,self.NBINS) )
                       self.countsAllClu = self.countsAllClu +  countsClu_i

                   #istogramma size clusters:
                   h_cluSizes_i, binsSizes_i = np.histogram(clu_sizes , bins = 100, range = (0,100) )
                   self.h_cluSizeAll=self.h_cluSizeAll+ h_cluSizes_i
    
       
        queue.put(self)
        
####################################
####################################
def     set_args(analizer,args):

      analizer.REBINXY=args.xyrebin
      analizer.APPLY_CLUSTERING=args.apply_clustering
      analizer.SAVE_EVENTLIST=args.save_eventlist
      analizer.myeps=args.myeps
      analizer.make_rawSpectrum=args.make_rawspectrum
      analizer.PIX_CUT_SIGMA=args.pix_cut_sigma # cut per pixel rumorosi
      analizer.CLU_CUT_SIGMA=args.clu_cut_sigma # clustering cut


def run_analyze(args):
    
    shots_path = args.inFilesPath
    bg_shots_path =args.bkgFilesPath
     
    print('\n==================================')
    print(' *** starting images analysis  ***\n')
    print("reading images from: ",shots_path )
    print("pedestals from: ",bg_shots_path)
    
    fileList=glob.glob( shots_path  + "/*.FIT")
    n_splits=args.n_jobs
    print("n. of parallel jobs: ",n_splits)
    

    countsAll2dRaw=None
    countsAll2dClu=None
    countsAll=None
    countsAllZeroSupp=None
    countsAllClu=None
    h_cluSizeAll=None
    w_all=None
    x_allClu=None
    y_allClu=None
    
    analizer_list=[]
    processes = []
    rets = []
    q = Queue()
    
    for i in range(1,n_splits+1):
    
        frames_block=int(len(fileList)/n_splits)
    
        low=(i-1)*frames_block
        if i==1:
            low=0
    
        up=i*frames_block
        if i==n_splits:
            up=len(fileList)
        
 
        block_fileList=fileList[low:up]
        
        analizer= analize_v2( block_fileList,bg_shots_path)
        set_args(analizer,args)
        if (i==1):
            print("\nanalize_v2 options:")
            print('   pixel_cut_sigma= ',analizer.PIX_CUT_SIGMA, '  (cut on pixels pedestal RMS)' )
            print('   clustering cut sigma= ',analizer.CLU_CUT_SIGMA, '  (cut on pixel value)'  )
            print('   apply clustering= ',analizer.APPLY_CLUSTERING)
            print('   dbscan eps=',analizer.myeps)
            print('   rebin xy=',analizer.REBINXY)
            print('   save eventList=',analizer.SAVE_EVENTLIST)
            print('   make_rawSpectrum=', analizer.make_rawSpectrum)
            print('==================================\n')
            
            
     
        p=Process(target=analizer.do_analyze ,args=(q,i,))
        p.start()
        processes.append(p)
        
    # set queue to get the data form porcess
    for process in processes:
        ret = q.get() # will block
        analizer_list.append(ret)

    #wait for all the sunprocedd to finish    
    for process in processes:
         process.join()
    print('\n... all data processed')
       


    # risommo tutte le componenti... (tutti i job devono aver finito!!)        
  
    for i in range(0,n_splits):
        if i==0:
            countsAll2dRaw=  analizer_list[i].countsAll2dRaw
            countsAll2dClu=  analizer_list[i].countsAll2dClu
            countsAll=  analizer_list[i].countsAll
            countsAllZeroSupp=  analizer_list[i].countsAllZeroSupp
            countsAllClu=  analizer_list[i].countsAllClu
            h_cluSizeAll= analizer_list[i].h_cluSizeAll
            w_all=analizer_list[i].w_all
            x_allClu=analizer_list[i].x_allClu
            y_allClu=analizer_list[i].y_allClu
            
        if i>0:
            countsAll2dRaw= countsAll2dRaw+ analizer_list[i].countsAll2dRaw
            countsAll2dClu= countsAll2dClu+ analizer_list[i].countsAll2dClu
            countsAll=countsAll +   analizer_list[i].countsAll
            countsAllZeroSupp= countsAllZeroSupp+ analizer_list[i].countsAllZeroSupp
            countsAllClu= countsAllClu+ analizer_list[i].countsAllClu
            h_cluSizeAll=h_cluSizeAll+analizer_list[i].h_cluSizeAll
            w_all=np.append(w_all,analizer_list[i].w_all)
            x_allClu=np.append(x_allClu, analizer_list[i].x_allClu)
            y_allClu=np.append(y_allClu, analizer_list[i].y_allClu)
            
            
            
        
     # plot immagini
    fig2, ax2 = plt.subplots()

    countsAll2dClu=  countsAll2dClu.T
    plt.imshow(countsAll2dClu, interpolation='nearest', origin='lower',  extent=[analizer_list[0].xedges[0], analizer_list[0].xedges[-1], analizer_list[0].yedges[0], analizer_list[0].yedges[-1]])
    plt.colorbar()
    plt.title('hit pixels (rebinned)')

    # plot immagine Raw
    fig3, ax3 = plt.subplots()
    countsAll2dRaw=  countsAll2dRaw.T
    plt.imshow(countsAll2dRaw, interpolation='nearest', origin='lower',    extent=[analizer_list[0].xedges[0], analizer_list[0].xedges[-1], analizer_list[0].yedges[0], analizer_list[0].yedges[-1]]  ) 
    plt.colorbar()
    plt.title('pixels>zero_suppression threshold')


    # plot spettro
    fig, h1 = plt.subplots()
    bins=analizer_list[0].bins
    if analizer_list[0].make_rawSpectrum==True:
        h1.hist(bins[:-1], bins = bins, weights = countsAll, histtype = 'step',label="raw")
    h1.hist(bins[:-1], bins = bins, weights = countsAllZeroSupp, histtype = 'step',label="pixel thresold")
    h1.hist(bins[:-1], bins = bins, weights = countsAllClu, histtype = 'step',label='CLUSTERING')
    plt.legend()
    plt.title('spectra')

    # plot spettro sizes
    fig5, h5 = plt.subplots()
    h5.hist(analizer_list[0].binsSize[:-1], bins =analizer_list[0].binsSize, weights = h_cluSizeAll , histtype = 'step',label='Cluster sizes')
    plt.legend()
    plt.title('CLU size')



    # save histos
    print('... saving histograms in:', shots_path )
    np.savez(shots_path+'spectrum_all_raw'+analizer_list[0].pixMask_suffix+'_parallel', counts = countsAll, bins = bins)
    np.savez(shots_path+'spectrum_all_ZeroSupp'+analizer_list[0].pixMask_suffix+analizer_list[0].cluCut_suffix+'_parallel', counts = countsAllZeroSupp, bins = bins)
    np.savez(shots_path+'spectrum_all_eps'+str(analizer_list[0].myeps)+analizer_list[0].pixMask_suffix+analizer_list[0].cluCut_suffix+'_parallel', counts = countsAllClu, bins = bins)
    np.savez(shots_path+'cluSizes_spectrum'+analizer_list[0].pixMask_suffix+'_parallel', counts = h_cluSizeAll , bins =analizer_list[0].binsSize )



    # save figures
    
    al.write_fitsImage(countsAll2dClu, shots_path+'imageCUL'+analizer_list[0].pixMask_suffix+analizer_list[0].cluCut_suffix+'_parallel.fits'  , overwrite = "False")
    al.write_fitsImage(countsAll2dRaw, shots_path+'imageRaw'+analizer_list[0].pixMask_suffix +'_parallel.fits'  , overwrite = "False")


    # salva vettori con event_list:
    if analizer_list[0].SAVE_EVENTLIST:
     outfileVectors= shots_path+'events_list'+analizer_list[0].pixMask_suffix+analizer_list[0].cluCut_suffix+'.npz'
     print('writing events in:',outfileVectors)
     al.save_vectors(outfileVectors, w_all, x_allClu, y_allClu)




      
if __name__ == '__main__':
    import time
    import argparse
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)

    parser.add_argument('-in','--inFilesPath', type=str,  help='path to FIT files', required=True)
    parser.add_argument('-bkg','--bkgFilesPath', type=str,  help='path to bkg files', required=True)
    parser.add_argument('--n_jobs', type=int,  help='n. of parallel jobs', required=False, default=3)
    parser.add_argument('--xyrebin', type=int,  help='rebin in image plot ', required=False, default=20)
    parser.add_argument('--pix_cut_sigma', type=int,  help='pixel ped rms cut in sigma units ', required=False, default=10)
    parser.add_argument('--clu_cut_sigma', type=int,  help='pixel cut in sigma units ', required=False, default=10)
    parser.add_argument('--no_clustering', dest='apply_clustering',  help='do NOT apply clustering', required=False, default=True,action='store_false')
    parser.add_argument('--no_eventlist', dest='save_eventlist',  help='do NOT save event list', required=False, default=True,  action='store_false'  )
    parser.add_argument('--make_rawspectrum',  help='make raw spectrum ', required=False, default=False, action='store_true' )
    parser.add_argument('--myeps', type=float,  help='DBSCAN eps', required=False, default=1.5)
   
    
    args = parser.parse_args()

    start = time.time()
    run_analyze(args)
    end = time.time()
    print('\n\nelapsed time: ',end - start,' [s]') 
    plt.show()

