import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import clustering_cmos 
import time
import zwoasi as asi
from astropy.io import fits

#inserire variabili globali
class aotr:
    def __init__(self, file_path, sample_size, WB_R, WB_B, EXPO, GAIN,bkg_folder_a, xyRebin, sigma, cluster, NoClustering, NoEvent, Raw, Eps):
        NBINS = 16384  # n.canali ADC (2^14)
        XBINS = 2822
        YBINS = 4144
        # Variabili passate all'analisi
        self.PIX_CUT_SIGMA = sigma
        self.CLU_CUT_SIGMA = cluster
        self.REBINXY = xyRebin
        self.APPLY_CLUSTERING = NoClustering
        self.SAVE_EVENTLIST = NoEvent
        self.myeps = Eps  # clustering DBSCAN
        self.pedfile = bkg_folder_a + '/mean_ped.fits'
        self.pedSigmafile = bkg_folder_a + '/std_ped.fits'
        # variabili passate alla camera e relative
        self.WB_R = WB_R
        self.WB_B = WB_B
        self.GAIN = GAIN
        self.EXPO = EXPO
        self.sample_size = sample_size

        #Variabili dell'analisi     DA RIVEDERE L'ORDINE

        self.pixMask_suffix='_pixCut'+str(PIX_CUT_SIGMA)+'sigma'
        self.cluCut_suffix='_CLUcut_'+str(CLU_CUT_SIGMA)+'sigma'
        self.xbins2d = int(XBINS / REBINXY)
        self.ybins2d = int(YBINS / REBINXY)
        self.x = []
        self.countsAll, self.bins = np.histogram(x, bins=2 * NBINS, range=(-NBINS, NBINS))
        self.countsAllZeroSupp, self.bins = np.histogram(x, bins=2 * NBINS, range=(-NBINS, NBINS))
        self.countsAllClu, self.bins = np.histogram(x, bins=2 * NBINS, range=(-NBINS, NBINS))
        self.h_cluSizeAll, self.binsSize = np.histogram(x, bins=100, range=(0, 100))
        self.x_allClu = np.empty(0)
        self.y_allClu = np.empty(0)
        self.w_all = np.empty(0)
        self.clusizes_all = np.empty(0)
        # creo histo2d vuoto:
        self.countsAll2dClu, self.xedges, self.yedges = np.histogram2d(x, x, bins=[xbins2d, ybins2d], range=[[0, XBINS], [0, YBINS]])
        self.countsAll2dRaw, self.xedgesRaw, self.yedgesRaw = np.histogram2d(x, x, bins=[xbins2d, ybins2d], range=[[0, XBINS], [0, YBINS]])



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

            for i in range (self.sample_size):

                # Ottieni i dati dell'immagine
                data = np.empty((2822, 4144), dtype=np.uint16)
                data = camera.capture()
                guiAnalyze(data)

        finally:
            # Arresta l'esposizione e rilascia la telecamera
            camera.stop_exposure()
            camera.close()

        ###########
        # plot immagine
        fig2, ax2 = plt.subplots()


        self.countsAll2dClu = countsAll2dClu.T
        plt.imshow(self.countsAll2dClu, interpolation='nearest', origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar()
        plt.title('hit pixels (rebinned)')

        # plot immagine Raw
        fig3, ax3 = plt.subplots()
        countsAll2dRaw = countsAll2dRaw.T
        plt.imshow(countsAll2dRaw, interpolation='nearest', origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar()
        plt.title('pixels>zero_suppression threshold')

        # plot spettro
        fig, h1 = plt.subplots()
        h1.hist(bins[:-1], bins=bins, weights=countsAll, histtype='step', label="raw")
        h1.hist(bins[:-1], bins=bins, weights=countsAllZeroSupp, histtype='step', label="pixel thresold")
        h1.hist(bins[:-1], bins=bins, weights=countsAllClu, histtype='step', label='CLUSTERING')
        plt.legend()
        plt.title('spectra')


        # plot spettro sizes
        fig5, h5 = plt.subplots()
        h5.hist(binsSize[:-1], bins=binsSize, weights=h_cluSizeAll, histtype='step', label='Cluster sizes')
        plt.legend()
        plt.title('CLU size')

        # save histos
        np.savez(file_path + 'spectrum_all_raw' + pixMask_suffix, counts=countsAll, bins=bins)
        np.savez(file_path + 'spectrum_all_ZeroSupp' + pixMask_suffix + cluCut_suffix, counts=countsAllZeroSupp, bins=bins)
        np.savez(file_path + 'spectrum_all_eps' + str(myeps) + pixMask_suffix + cluCut_suffix, counts=countsAllClu,
                 bins=bins)
        np.savez(file_path + 'cluSizes_spectrum' + pixMask_suffix, counts=h_cluSizeAll, bins=binsSize)

        # save figures
        al.write_fitsImage(countsAll2dClu, file_path + 'imageCUL' + pixMask_suffix + cluCut_suffix + '.fits',
                           overwrite="False")
        # al.write_fitsImage(image_SW, shots_path+'imageSUM'+pixMask_suffix +'.fits'  , overwrite = "False")
        al.write_fitsImage(countsAll2dRaw, file_path + 'imageRaw' + pixMask_suffix + '.fits', overwrite="False")

        # salva vettori con event_list:
        if SAVE_EVENTLIST:
            outfileVectors = file_path + 'events_list' + pixMask_suffix + cluCut_suffix + '_v2.npz'
            print('writing events in:', outfileVectors)
            # al.save_vectors(outfileVectors, w_all, x_allClu, y_allClu,clusizes_all)
            np.savez(outfileVectors, w=w_all, x_pix=x_allClu, y_pix=y_allClu, sizes=clusizes_all)

        plt.show()



    def guiAnalyze(foto):

        xbins2d=int(XBINS/REBINXY)
        ybins2d=int(YBINS/REBINXY)

        # inizio analisi...
        # leggo files pedestal (mean e rms)
        mean_ped = al.read_image(pedfile)
        rms_ped = al.read_image(pedSigmafile)

        rms_pedCut=np.mean(rms_ped)+PIX_CUT_SIGMA*np.std(rms_ped)
        # MASCHERA PIXEL RUMOROSI

        mySigmaMask=np.where( (rms_ped>rms_pedCut) )

        # read image:
        image_data = foto/4.
        # subtract pedestal:
        image_data = image_data -  mean_ped #

        #applica maschera
        image_data[mySigmaMask]=0 # maschero tutti i pixel con RMS pedestal > soglia

        #image_SW = image_SW + image_data
        flat_image = image_data.flatten()

        # spettro "raw"
        counts_i, bins_i = np.histogram(flat_image,  bins = 2*NBINS, range = (-NBINS,NBINS) )
        countsAll = countsAll + counts_i



        #################
        #ZERO SUPPRESSION
        # applico selezione su carica dei pixel
        supp_coords, supp_weights=al.select_pixels_RMS(image_data, rms_ped, CLU_CUT_SIGMA)

        # salvo pixel che sopravvivono alla selezione:
        zeroSupp_trasposta= supp_coords

        #istogramma 2d immagine raw dopo zero suppression:
        counts2dRaw,  xedgesRaw, yedgesRaw= np.histogram2d(zeroSupp_trasposta[0],zeroSupp_trasposta[1],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
        self.countsAll2dRaw=countsAll2dRaw+counts2dRaw

        #spettro dopo zeroSuppression
        countsZeroSupp_i, bins_i = np.histogram( supp_weights,  bins = 2*NBINS, range = (-NBINS,NBINS) )
        self.countsAllZeroSupp =countsAllZeroSupp  +  countsZeroSupp_i
        #CLUSTERING
        if APPLY_CLUSTERING:

            # test clustering.... # uso v2 per avere anche le posizioni
            w_clusterAll, clu_coordsAll, clu_sizes, clu_baryCoords    =clustering_cmos.clustering_v3(np.transpose(supp_coords),supp_weights,myeps=myeps)
            cluBary_trasposta= clu_baryCoords.transpose()

            if SAVE_EVENTLIST:
                self.x_allClu=np.append(x_allClu,cluBary_trasposta[0])
                self.y_allClu=np.append(y_allClu,cluBary_trasposta[1])
                self.w_all=np.append(w_all,w_clusterAll)
                self.clusizes_all=np.append(clusizes_all,clu_sizes)
            # istogramma 2d dopo clustering solo baricentri!!!!
            counts2dClu,  xedges, yedges= np.histogram2d(cluBary_trasposta[0],cluBary_trasposta[1],bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
            self.countsAll2dClu=countsAll2dClu+ counts2dClu

            # istogramma spettro dopo il clustering
            size_mask=np.where(clu_sizes>0) # select all clusters!!!!
            countsClu_i, bins_i = np.histogram(  w_clusterAll[size_mask], bins = 2*NBINS, range = (-NBINS,NBINS) )
            self.countsAllClu = countsAllClu +  countsClu_i

            #istogramma size clusters:
            h_cluSizes_i, binsSizes_i = np.histogram(clu_sizes , bins = 100, range = (0,100) )
            self.h_cluSizeAll=h_cluSizeAll+ h_cluSizes_i


