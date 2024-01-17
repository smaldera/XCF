import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import clustering_cmos 
import time
import zwoasi as asi
from tqdm import  tqdm
from astropy.io import fits

#inserire variabili globali
class aotr:
    def __init__(self, file_path, sample_size, WB_R, WB_B, EXPO, GAIN,bkg_folder_a, xyRebin, sigma, cluster, NoClustering, NoEvent, Raw, Eps):
        self.NBINS = 16384  # n.canali ADC (2^14)
        self.XBINS = 2822
        self.YBINS = 4144
        # Variabili passate all'analisi
        self.PIX_CUT_SIGMA = sigma
        self.CLU_CUT_SIGMA = cluster
        self.REBINXY = xyRebin
        self.APPLY_CLUSTERING = NoClustering
        self.SAVE_EVENTLIST = NoEvent
        self.myeps = Eps  # clustering DBSCAN
        self.pedfile = bkg_folder_a + '/mean_ped.fits'
        self.pedSigmafile = bkg_folder_a + '/std_ped.fits'
        self.mean_ped = al.read_image(self.pedfile)
        self.rms_ped = al.read_image(self.pedSigmafile)


        # variabili passate alla camera e relative
        self.WB_R = WB_R
        self.WB_B = WB_B
        self.GAIN = GAIN
        self.EXPO = EXPO
        self.sample_size = sample_size

        #Variabili dell'analisi     DA RIVEDERE L'ORDINE
        self.xbins2d = int(self.XBINS / self.REBINXY)
        self.ybins2d = int(self.YBINS / self.REBINXY)
        self.pixMask_suffix='_pixCut'+str(self.PIX_CUT_SIGMA)+'sigma'
        self.cluCut_suffix='_CLUcut_'+str(self.CLU_CUT_SIGMA)+'sigma'

        self.x = []
        self.file_path = file_path
        self.countsAll, self.bins = np.histogram(self.x, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
        self.countsAllZeroSupp, self.bins = np.histogram(self.x, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
        self.countsAllClu, self.bins = np.histogram(self.x, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
        self.h_cluSizeAll, self.binsSize = np.histogram(self.x, bins=100, range=(0, 100))
        self.x_allClu = np.empty(0)
        self.y_allClu = np.empty(0)
        self.w_all = np.empty(0)
        self.clusizes_all = np.empty(0)
        # creo histo2d vuoto:
        self.countsAll2dClu, self.xedges, self.yedges = np.histogram2d(self.x, self.x, bins=[self.xbins2d, self.ybins2d], range=[[0, self.XBINS], [0, self.YBINS]])
        self.countsAll2dRaw, self.xedgesRaw, self.yedgesRaw = np.histogram2d(self.x, self.x, bins=[self.xbins2d, self.ybins2d], range=[[0, self.XBINS], [0, self.YBINS]])








    def CaptureAnalyze(self, camera):

        #Setting up variables


        try:
            # Use minimum USB bandwidth permitted
            camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, camera.get_controls()['BandWidth']['MinValue'])

            # Set some sensible defaults. They will need adjusting depending upon
            camera.disable_dark_subtract()
            camera.set_control_value(asi.ASI_GAMMA, 50)
            camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
            camera.set_control_value(asi.ASI_FLIP, 0)
            camera.set_control_value(asi.ASI_GAIN, self.GAIN)
            camera.set_control_value(asi.ASI_WB_B, self.WB_B)
            camera.set_control_value(asi.ASI_WB_R, self.WB_R)
            camera.set_control_value(asi.ASI_EXPOSURE, self.EXPO)
            camera.set_image_type(asi.ASI_IMG_RAW16)

            try:
                # Force any single exposure to be halted
                camera.stop_video_capture()
                camera.stop_exposure()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                pass


            custom_style = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

            for i in tqdm(range (self.sample_size)):


                # Ottieni i dati dell'immagine
                data = np.empty((2822, 4144), dtype=np.uint16)
                data = camera.capture()

                rms_pedCut=np.mean(self.rms_ped)+self.PIX_CUT_SIGMA*np.std(self.rms_ped)
                # MASCHERA PIXEL RUMOROSI

                mySigmaMask=np.where( (self.rms_ped>rms_pedCut) )
        
                # read image:
                image_data = data/4.
                # subtract pedestal:
                image_data = image_data -  self.mean_ped #

                #applica maschera
                image_data[mySigmaMask]=0 # maschero tutti i pixel con RMS pedestal > soglia

                #image_SW = image_SW + image_data
                flat_image = image_data.flatten()

                # spettro "raw"
                counts_i, bins_i = np.histogram(flat_image,  bins = 2*self.NBINS, range = (-self.NBINS,self.NBINS) )
                self.countsAll = self.countsAll + counts_i



                #################
                #ZERO SUPPRESSION
                # applico selezione su carica dei pixel
                supp_coords, supp_weights=al.select_pixels_RMS(image_data, self.rms_ped, self.CLU_CUT_SIGMA)

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

                    # test clustering.... # uso v2 per avere anche le posizioni
                    self.w_clusterAll, self.clu_coordsAll, clu_sizes, clu_baryCoords    =clustering_cmos.clustering_v3(np.transpose(supp_coords),supp_weights,myeps=self.myeps)
                    cluBary_trasposta= clu_baryCoords.transpose()

                    if self.SAVE_EVENTLIST:
                        self.x_allClu=np.append(self.x_allClu,cluBary_trasposta[0])
                        self.y_allClu=np.append(self.y_allClu,cluBary_trasposta[1])
                        self.w_all=np.append(self.w_all,self.w_clusterAll)
                        self.clusizes_all=np.append(self.clusizes_all,clu_sizes)
                    # istogramma 2d dopo clustering solo baricentri!!!!
                    counts2dClu,  xedges, yedges= np.histogram2d(cluBary_trasposta[0],cluBary_trasposta[1],bins=[self.xbins2d, self.ybins2d ],range=[[0,self.XBINS],[0,self.YBINS]])
                    self.countsAll2dClu=self.countsAll2dClu+ counts2dClu

                    # istogramma spettro dopo il clustering
                    size_mask=np.where(clu_sizes>0) # select all clusters!!!!
                    countsClu_i, bins_i = np.histogram(  self.w_clusterAll[size_mask], bins = 2*self.NBINS, range = (-self.NBINS,self.NBINS) )
                    self.countsAllClu = self.countsAllClu +  countsClu_i

                    #istogramma size clusters:
                    h_cluSizes_i, binsSizes_i = np.histogram(clu_sizes , bins = 100, range = (0,100) )
                    self.h_cluSizeAll= self.h_cluSizeAll+ h_cluSizes_i





        finally:
            # Arresta l'esposizione e rilascia la telecamera
            camera.stop_exposure()
            camera.close()

        ###########
        # plot immagine
        fig2, ax2 = plt.subplots()


        self.countsAll2dClu = self.countsAll2dClu.T
        plt.imshow(self.countsAll2dClu, interpolation='nearest', origin='lower',
                   extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
        plt.colorbar()
        plt.title('hit pixels (rebinned)')

        # plot immagine Raw
        fig3, ax3 = plt.subplots()
        self.countsAll2dRaw = self.countsAll2dRaw.T
        plt.imshow(self.countsAll2dRaw, interpolation='nearest', origin='lower',
                   extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
        plt.colorbar()
        plt.title('pixels>zero_suppression threshold')

        # plot spettro
        fig, h1 = plt.subplots()
        h1.hist(self.bins[:-1], bins=self.bins, weights=self.countsAll, histtype='step', label="raw")
        h1.hist(self.bins[:-1], bins=self.bins, weights=self.countsAllZeroSupp, histtype='step', label="pixel thresold")
        h1.hist(self.bins[:-1], bins=self.bins, weights=self.countsAllClu, histtype='step', label='CLUSTERING')
        plt.legend()
        plt.title('spectra')


        # plot spettro sizes
        fig5, h5 = plt.subplots()
        h5.hist(self.binsSize[:-1], bins=self.binsSize, weights=self.h_cluSizeAll, histtype='step', label='Cluster sizes')
        plt.legend()
        plt.title('CLU size')

        # save histos
        np.savez(self.file_path + 'spectrum_all_raw' + self.pixMask_suffix, counts=self.countsAll, bins=self.bins)
        np.savez(self.file_path + 'spectrum_all_ZeroSupp' + self.pixMask_suffix + self.cluCut_suffix, counts=self.countsAllZeroSupp, bins=self.bins)
        np.savez(self.file_path + 'spectrum_all_eps' + str(self.myeps) + self.pixMask_suffix + self.cluCut_suffix, counts=self.countsAllClu,
                 bins=self.bins)
        np.savez(self.file_path + 'cluSizes_spectrum' + self.pixMask_suffix, counts=self.h_cluSizeAll, bins=self.binsSize)

        # save figures
        al.write_fitsImage(self.countsAll2dClu, self.file_path + 'imageCUL' + self.pixMask_suffix + self.cluCut_suffix + '.fits',
                           overwrite="False")
        # al.write_fitsImage(image_SW, shots_path+'imageSUM'+pixMask_suffix +'.fits'  , overwrite = "False")
        al.write_fitsImage(self.countsAll2dRaw, self.file_path + 'imageRaw' + self.pixMask_suffix + '.fits', overwrite="False")

        # salva vettori con event_list:
        if self.SAVE_EVENTLIST:
            outfileVectors = self.file_path + 'events_list' + self.pixMask_suffix + self.cluCut_suffix + '_v2.npz'
            print('writing events in:', outfileVectors)
            # al.save_vectors(outfileVectors, w_all, x_allClu, y_allClu,clusizes_all)
            np.savez(outfileVectors, w=self.w_all, x_pix=self.x_allClu, y_pix=self.y_allClu, sizes=self.clusizes_all)

        plt.show()



