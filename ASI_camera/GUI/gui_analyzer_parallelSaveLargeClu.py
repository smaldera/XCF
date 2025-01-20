from cmos_pedestal import checkup
import numpy as np
from matplotlib import pyplot as plt

#import sys
import utils_v2 as al
import clustering_cmos 
#import zwoasi as asi
from tqdm import  tqdm
from astropy.io import fits
import multiprocessing
import FreeSimpleGUI as sg
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# CMOS Energy calibration parameters
calP1= 0.00321327
calP0=-0.0032013
LIVE_PLOTS=False

class aotr2:


    def __init__(self, file_path, sample_size, WB_R, WB_B, EXPO, GAIN,bkg_folder_a, xyRebin, sigma, cluster, NoClustering, NoEvent, Raw, Eps, num ,leng,bunch):
        self.NBINS = 16384  # ADC (14 bit)
        self.XBINS = 2822
        self.YBINS = 4144
        self.length = leng
        self.PIX_CUT_SIGMA = sigma
        self.CLU_CUT_SIGMA = cluster
        self.REBINXY = xyRebin
        self.APPLY_CLUSTERING = NoClustering
        self.SAVE_EVENTLIST = NoEvent
        self.myeps = Eps  # DBSCAN
        self.pedfile = bkg_folder_a + '/mean_ped.fits'
        self.pedSigmafile = bkg_folder_a + '/std_ped.fits'
        self.mean_ped = al.read_image(self.pedfile)
        self.rms_ped = al.read_image(self.pedSigmafile)
        self.num = num
        self.bunch = bunch

        # Camera Variables
        self.WB_R = WB_R
        self.WB_B = WB_B
        self.GAIN = GAIN
        self.EXPO = EXPO
        self.sample_size = sample_size

        # Analysis Variables
        self.xbins2d = int(self.XBINS / self.REBINXY)
        self.ybins2d = int(self.YBINS / self.REBINXY)
        self.pixMask_suffix='_pixCut'+str(self.PIX_CUT_SIGMA)+'sigma'
        self.cluCut_suffix='_CLUcut_'+str(self.CLU_CUT_SIGMA)+'sigma'
        self.list = []

        self.x = []
        self.file_path = file_path + '/'
        self.countsAll, self.bins = np.histogram(self.x, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
        self.countsAllZeroSupp, self.bins = np.histogram(self.x, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
        self.countsAllClu, self.bins = np.histogram(self.x, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
        self.h_cluSizeAll, self.binsSize = np.histogram(self.x, bins=100, range=(0, 100))
        self.x_allClu = np.empty(0)
        self.y_allClu = np.empty(0)
        self.w_all = np.empty(0)
        self.clusizes_all = np.empty(0)
        self.make_rawSpectrum = False
        self.countsAll2dClu, self.xedges, self.yedges = np.histogram2d(self.x, self.x, bins=[self.xbins2d, self.ybins2d], range=[[0, self.XBINS], [0, self.YBINS]])
        self.countsAll2dRaw, self.xedgesRaw, self.yedgesRaw = np.histogram2d(self.x, self.x, bins=[self.xbins2d, self.ybins2d], range=[[0, self.XBINS], [0, self.YBINS]])



    def Analizza(self, data_queue,id,data_buffer,lock):

        print ("... starting analizza")
        n_images=0  #n.images saved when cluSize>15
        self.reset_allVariables()
        progress_bar2 = tqdm(total=(self.sample_size/self.num), desc="Analizzatore_" + str(id), colour='green', position=self.num+id)
        if id==0:
            layout = [
		    [sg.Text('Progresso:', size=(10, 1)),sg.ProgressBar((self.sample_size/self.num), orientation='h', size=(20, 20), key='progress')],]
            window_progress = sg.Window('Data cruncher number: ' + str(id), layout, finalize=True)
            progress_bar = window_progress['progress']
        i=1
        rms_pedCut = np.mean(self.rms_ped) + self.PIX_CUT_SIGMA * np.std(self.rms_ped)   
        mySigmaMask = np.where((self.rms_ped > rms_pedCut))                              

        while True:
            data= data_queue.get()
            if data is None:
                progress_bar2.close()
                if (id==0):
                    window_progress.Close()
                break

            #print("DEBUG=> analizza: data.shape=",data.shape)
           
            image_data = data / 4.
            image_data = image_data - self.mean_ped
            image_data[mySigmaMask] = 0
            flat_image = image_data.flatten()

            counts_i, _ = np.histogram(flat_image, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
           
            supp_coords, supp_weights = al.select_pixels_RMS(image_data, self.rms_ped, self.CLU_CUT_SIGMA)
            if len(supp_weights)==0:
                print ("empty event",i,"... skipping")
                continue
            
            zeroSupp_trasposta = supp_coords
            counts2dRaw, _, _ = np.histogram2d(zeroSupp_trasposta[0], zeroSupp_trasposta[1],
                                                               bins=[self.xbins2d, self.ybins2d],
                                                               range=[[0, self.XBINS], [0, self.YBINS]])
            self.countsAll2dRaw = self.countsAll2dRaw + counts2dRaw

            countsZeroSupp_i, _ = np.histogram(supp_weights, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
           

            if self.APPLY_CLUSTERING:
                self.w_clusterAll, self.clu_coordsAll, clu_sizes, clu_baryCoords = clustering_cmos.clustering_v3(np.transpose(supp_coords), supp_weights, myeps=self.myeps)
                cluBary_trasposta = clu_baryCoords.transpose()
                counts2dClu, _, _ = np.histogram2d(cluBary_trasposta[0], cluBary_trasposta[1],bins=[self.xbins2d, self.ybins2d],range=[[0, self.XBINS], [0, self.YBINS]])
                countsClu_i, _ = np.histogram(self.w_clusterAll, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
                h_cluSizes_i, _ = np.histogram(clu_sizes, bins=100, range=(0, 100))                               
                self.h_cluSizeAll = self.h_cluSizeAll + h_cluSizes_i

                if (max(clu_sizes)>9) and  (n_images<1000) : 
                     #salvo immagine
                     n_images+=1 
                     nomefile=self.file_path+'/img_cluSize10_'+str(n_images)+'.fits'
                     header = fits.Header()
                     header['DATE-OBS'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
                     #Saving Image.FITS
                     # print("DEBUG11: before saving.. data.shape=",data.shape)
                     hdu = fits.PrimaryHDU(data, header=header)
                     hdulist = fits.HDUList([hdu])
                     hdulist.writeto(nomefile, overwrite=True)
                     
                
            progress_bar2.update(1)

            if id==0:
                progress_bar.UpdateBar(i)
            i += 1

            with lock: 
                data_buffer[0] += counts2dRaw
                data_buffer[2] += counts_i
                data_buffer[3] += countsZeroSupp_i
                if self.APPLY_CLUSTERING:
                    data_buffer[1] += counts2dClu
                    data_buffer[4] += countsClu_i
                    data_buffer[5] += h_cluSizes_i
                    if self.SAVE_EVENTLIST:
                        data_buffer[6] = np.append(data_buffer[6] , self.w_clusterAll)
                        data_buffer[7] = np.append(data_buffer[7] , cluBary_trasposta[0])
                        data_buffer[8] = np.append(data_buffer[8] , cluBary_trasposta[1])
                        data_buffer[9] = np.append(data_buffer[9] , clu_sizes)



    def initialize_camera(self):
        import zwoasi as asi
        camera = checkup()
       
        #Use minimum USB bandwidth permitted
        camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 95)
        camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, True)
        camera.disable_dark_subtract()
        camera.set_control_value(asi.ASI_GAMMA, 50)
        camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
        camera.set_control_value(asi.ASI_FLIP, 0)
        camera.set_control_value(asi.ASI_GAIN, self.GAIN)
        camera.set_control_value(asi.ASI_WB_B, self.WB_B)
        camera.set_control_value(asi.ASI_WB_R, self.WB_R)
        camera.set_control_value(asi.ASI_EXPOSURE, self.EXPO)
        camera.set_image_type(asi.ASI_IMG_RAW16)
        timeout = (camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 10500
        camera.default_timeout = timeout


        return camera
           
             

    def CaptureAnalyze(self):
        import time
        import zwoasi as asi
        
        print("... starting Capture and Analize!!! ")
       # camera = checkup()
        camera=self.initialize_camera()
        lock = multiprocessing.Lock() # Data can't be simoultanously analysed by 2 or more processes
        
        # Buffers
        manager =multiprocessing.Manager()
        data_buffer = manager.list(range(10))
        data_buffer[0] = self.countsAll2dRaw
        data_buffer[1] = self.countsAll2dClu
        data_buffer[2] = self.countsAll
        data_buffer[3] = self.countsAllZeroSupp
        data_buffer[4] = self.countsAllClu
        data_buffer[5] = self.h_cluSizeAll
        data_buffer[6] = self.w_all
        data_buffer[7] = self.x_allClu
        data_buffer[8] = self.y_allClu
        data_buffer[9] = self.clusizes_all
        
        data_queue = multiprocessing.Queue()

        numero_analizzatori = self.num
        processi = []
        for i in range(numero_analizzatori):
            processo = multiprocessing.Process(target=self.Analizza, args= ( data_queue, i,data_buffer, lock))
            processi.append(processo)
        for processo in processi:
            processo.start()


       
        layout_capture = [
            [sg.Text('Cattura in corso:', size=(15, 1)), sg.ProgressBar(self.sample_size, orientation='h', size=(20, 20), key='progress_capture')],
        ]
        window_capture = sg.Window('Cattura in corso', layout_capture, finalize=True)
        progress_bar_capture = window_capture['progress_capture']

        # --------------------------------------CAMERA--------------------------------------
        try:
            
                    
            temp=[]
            mytime=[]
            k=1
            plt.ion()
            if LIVE_PLOTS==True:
                fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(11,7), constrained_layout=True )
                plt.subplots_adjust(wspace=0.4)
                fig3.canvas.manager.window.wm_geometry("+%d+%d" % (2, 2))
                ax3[1].set_xlim([0,12])  #starting x limits
                div = make_axes_locatable(ax3[0])
                cax = div.append_axes('right', '5%', '5%')
            
            
            for i in tqdm(range (self.sample_size), desc='acquiring data', colour='green'):

                # Taking snaps every 100 images
                if(i%100==0): 
                    t=camera.get_control_value(asi.ASI_TEMPERATURE)[0]/10.
                    temp.append(t)
                    mytime.append(datetime.utcnow().timestamp())
                    np.savez('temps.npz',time=mytime, temp=temp)

                    while (t>50): 
                        # Stop video capture and wait for t to drop below 50
                        camera.stop_video_capture()
                        time.sleep(900)
                        t=camera.get_control_value(asi.ASI_TEMPERATURE)[0]/10.
                        temp.append(t)
                        mytime.append(datetime.utcnow().timestamp())
                        np.savez('temps.npz',time=mytime, temp=temp)
                        camera.start_video_capture()

                # Start capturing the first time
                if i == 0 :
                    camera.start_video_capture()

                # Stop capture so that analyzer can do its things
                while data_queue.qsize() > self.length:
                    camera.stop_video_capture()
                    print("Waiting for analyzer to catch up")
                    time.sleep(15)
                    camera.start_video_capture()

                while True:
                    try:
                        data = np.empty((2822, 4144), dtype=np.uint16)
                        data = camera.capture_video_frame()
                        #print("DEBUG=> captureAnalyze: data.shape=",data.shape)
                        data_queue.put(data)
                        break
                    except Exception as e:
                      #  camera.stop_video_capture()
                      
                        print("There are troubles in CaptureAnalyze True:" , e, ", frame = ", i)
                        print("stopping video capture...")
                        camera.stop_video_capture()
                        time.sleep(10)
                        camera=self.initialize_camera()
                        print("restarting video capture...")
                        camera.start_video_capture()
                        print("... done")
                        
                progress_bar_capture.UpdateBar(i)

                # ---------------------------------LIVE  PLOTS------------------------------------
                if LIVE_PLOTS==True:
                    if i%20 == 0:
                        self.recover_data(data_buffer)
                        self.livePlots(ax3,fig3,cax)


                # ------------------SAVINGS-------------------
                if ((i) >= ((self.bunch)*k)):
                    n_event_analized = i - data_queue.qsize()
                    k+=1
                    self.recover_data(data_buffer)
                    file = open(self.file_path+ 'numero_eventi', 'w')
                    file.write(str(n_event_analized))

                    np.savez(self.file_path + 'spectrum_all_raw' + self.pixMask_suffix, counts=self.countsAll,
                             bins=self.bins)
                    np.savez(self.file_path + 'spectrum_all_ZeroSupp' + self.pixMask_suffix + self.cluCut_suffix,
                             counts=self.countsAllZeroSupp, bins=self.bins)
                    np.savez(self.file_path + 'spectrum_all_eps' + str(
                        self.myeps) + self.pixMask_suffix + self.cluCut_suffix, counts=self.countsAllClu,
                             bins=self.bins)
                    np.savez(self.file_path + 'cluSizes_spectrum' + self.pixMask_suffix, counts=self.h_cluSizeAll,
                             bins=self.binsSize)

                    # Save figures
                    al.write_fitsImage(self.countsAll2dClu,
                                       self.file_path + 'imageCUL' + self.pixMask_suffix + self.cluCut_suffix + '.fits',
                                       overwrite="False")
                    al.write_fitsImage(self.countsAll2dRaw,
                                       self.file_path + 'imageRaw' + self.pixMask_suffix + '.fits',
                                       overwrite="False")
                    # Save eventilsts arrays
                    if self.SAVE_EVENTLIST:
                        outfileVectors = self.file_path + 'events_list' + self.pixMask_suffix + self.cluCut_suffix + '_v2.npz'
                        np.savez(outfileVectors, w=self.w_all, x_pix=self.x_allClu, y_pix=self.y_allClu,
                                 sizes=self.clusizes_all)


            window_capture.Close()
            for asd in range(0,self.num):
                data_queue.put(None)
                asd+=1
                
        finally:
            camera.stop_exposure()
            camera.close()

        analizer_list = []
        for processo in processi:
            processo.join()
        #if  LIVE_PLOTS==True:
        #    plt.close(fig3)
        
        
        plt.ioff()

        self.recover_data(data_buffer)
        file = open(self.file_path+ 'numero_eventi', 'w')
        file.write(str(self.sample_size))        
        
        self.final_plots()


    def recover_data(self, data_buffer):
        self.countsAll2dRaw = data_buffer[0]
        self.countsAll2dClu = data_buffer[1]
        self.countsAll = data_buffer[2]
        self.countsAllZeroSupp = data_buffer[3]
        self.countsAllClu = data_buffer[4]
        self.h_cluSizeAll = data_buffer[5]
        self.w_all = data_buffer[6]
        self.x_allClu = data_buffer[7]
        self.y_allClu = data_buffer[8]
        self.clusizes_all = data_buffer[9]

    
    def reset_allVariables(self):
        x = []

        self.countsAll2dClu = np.array(x)
        self.xedges = np.array(x)
        self.yedges = np.array(x)
        self.countsAll = np.array(x)
        self.bins = np.array(x)
        self.countsAllZeroSupp = np.array(x)
        self.countsAllClu = np.array(x)
        self.h_cluSizeAll = np.array(x)
        self.countsAll2dRaw = np.array(x)
        self.xedgesRaw = np.array(x)
        self.yedgesRaw = np.array(x)
        self.countsAll, self.bins = np.histogram(x, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
        self.countsAllZeroSupp, self.bins = np.histogram(x, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
        self.countsAllClu, self.bins = np.histogram(x, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
        self.h_cluSizeAll, self.binsSize = np.histogram(x, bins=100, range=(0, 100))
        self.countsAll2dClu, self.xedges, self.yedges = np.histogram2d(x, x, bins=[self.xbins2d, self.ybins2d],
                                                                       range=[[0, self.XBINS], [0, self.YBINS]])
        self.countsAll2dRaw, self.xedgesRaw, self.yedgesRaw = np.histogram2d(x, x, bins=[self.xbins2d, self.ybins2d],
                                                                             range=[[0, self.XBINS], [0, self.YBINS]])
        self.x_allClu = np.empty(0)
        self.y_allClu = np.empty(0)
        self.w_all = np.empty(0)
        self.clusizes_all = np.empty(0)
        




    def livePlots(self, ax3,fig,cax):
           All2dRaw = self.countsAll2dRaw.T
           old_xLim=ax3[1].get_xlim()
           
           ax3[0].cla()
           #im=ax3[0].imshow(np.log10(All2dRaw), interpolation='nearest', origin='lower', extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
           im=ax3[0].imshow(All2dRaw, interpolation='nearest', origin='lower', extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
           self.bins = np.linspace(-self.NBINS, self.NBINS, 2*self.NBINS+1)
           self.bins = calP0 + calP1 * self.bins


           
           ax3[1].cla()
           cax.cla()
           ax3[1].hist(self.bins[:-1], bins=self.bins, weights=self.countsAllZeroSupp, histtype='step', label="pixel thresold", color = "green")
           ax3[1].hist(self.bins[:-1], bins=self.bins, weights=self.countsAllClu, histtype='step', label='CLUSTERING', color = "red")
                    
           ax3[0].set_title("Image Raw", fontsize = 16)
           ax3[0].set_xlabel("X [pix]", fontsize = 12)
           ax3[0].set_ylabel("Y [pix]", fontsize = 12)
          
           cb=fig.colorbar(im,cax=cax, orientation='vertical')
           
           
           ax3[1].set_xlabel("Energy [keV]", fontsize = 12)
           ax3[1].set_ylabel("log10(Counts)", fontsize = 12)
           ax3[1].set_xlim(old_xLim)
           ax3[1].set_yscale('log')
           ax3[1].set_title("Spectrum", fontsize = 16)
           ax3[1].legend()
          
         

    def final_plots(self):
        
        fig6, ax6 = plt.subplots(nrows=2, ncols=2, figsize=(11,7) )
        self.countsAll2dClu = self.countsAll2dClu.T
        im=ax6[0,0].imshow(np.log10(self.countsAll2dClu), interpolation='nearest', origin='lower', extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
        #plt.colorbar(ax6[0,0])
        
        ax6[0,0].set_title('hit pixels clustering  (rebinned)')
        fig6.colorbar(im,ax=ax6[0,0], orientation='vertical')

        # plot immagine Raw
        self.countsAll2dRaw = self.countsAll2dRaw.T
        im2=ax6[1,0].imshow(self.countsAll2dRaw, interpolation='nearest', origin='lower',  extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
        #plt.colorbar()
        ax6[1,0].set_title('pixels >zero_suppression threshold')
        fig6.colorbar(im2,ax=ax6[1,0], orientation='vertical')

        
        # plot spettro
        self.bins = np.linspace(-self.NBINS, self.NBINS, 2*self.NBINS+1)
        self.bins = calP0 + calP1 * self.bins
        ax6[1,1].hist(self.bins[:-1], bins=self.bins, weights=self.countsAll, histtype='step', label="raw")
        ax6[1,1].hist(self.bins[:-1], bins=self.bins, weights=self.countsAllZeroSupp, histtype='step', label="pixel thresold")
        ax6[1,1].hist(self.bins[:-1], bins=self.bins, weights=self.countsAllClu, histtype='step', label='CLUSTERING')
        plt.legend()
        ax6[1,1].set_title('spectra', fontsize = 12)
        ax6[1,1].set_xlabel("Energy [keV]", fontsize = 12)
        ax6[1,1].set_yscale('log')
        ax6[1,1].set_xlim([0,12])
        
        # plot spettro sizes
        ax6[0,1].hist(self.binsSize[:-1], bins=self.binsSize, weights=self.h_cluSizeAll, histtype='step', label='Cluster sizes')
        plt.legend()
        ax6[0,1].set_title('CLU size')
        ax6[0,1].set_yscale('log')
        ax6[0,1].set_xlim([0,10])
       

        # save histos
        np.savez(self.file_path + 'spectrum_all_raw' + self.pixMask_suffix, counts=self.countsAll, bins=self.bins)
        np.savez(self.file_path + 'spectrum_all_ZeroSupp' + self.pixMask_suffix + self.cluCut_suffix, counts=self.countsAllZeroSupp, bins=self.bins)
        np.savez(self.file_path + 'spectrum_all_eps' + str(self.myeps) + self.pixMask_suffix + self.cluCut_suffix, counts=self.countsAllClu,  bins=self.bins)
        np.savez(self.file_path + 'cluSizes_spectrum' + self.pixMask_suffix, counts=self.h_cluSizeAll, bins=self.binsSize)

        # save figures
        al.write_fitsImage(self.countsAll2dClu, self.file_path + 'imageCUL' + self.pixMask_suffix + self.cluCut_suffix + '.fits',
                           overwrite="False")
        al.write_fitsImage(self.countsAll2dRaw, self.file_path + 'imageRaw' + self.pixMask_suffix + '.fits', overwrite="False")

        print ('SAVE_EVENTLIST=',self.SAVE_EVENTLIST)
        if self.SAVE_EVENTLIST:
            outfileVectors = self.file_path + 'events_list' + self.pixMask_suffix + self.cluCut_suffix + '_v2.npz'
            np.savez(outfileVectors, w=self.w_all, x_pix=self.x_allClu, y_pix=self.y_allClu, sizes=self.clusizes_all)

        plt.show()
