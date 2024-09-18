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
import multiprocessing
import FreeSimpleGUI as sg
from datetime import datetime
from multiprocessing import Manager, Lock
from cmos_pedestal import checkup


# CMOS Energy calibration parameters
calP1= 0.00321327
calP0=-0.0032013

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
        self.reset_allVariables()
        progress_bar2 = tqdm(total=(self.sample_size/self.num), desc="Analizzatore_" + str(id), colour='green', position=self.num+id)
        if id==0:
            layout = [
		    [sg.Text('Progresso:', size=(10, 1)),sg.ProgressBar((self.sample_size/self.num), orientation='h', size=(20, 20), key='progress')],]
            window_progress = sg.Window('Data cruncher number: ' + str(id), layout, finalize=True)
            progress_bar = window_progress['progress']
        i=1

        while True:
            data= data_queue.get()
            if data is None:
                progress_bar2.close()
                if (id==0):
                    window_progress.Close()
                break
                
            rms_pedCut = np.mean(self.rms_ped) + self.PIX_CUT_SIGMA * np.std(self.rms_ped)
            mySigmaMask = np.where((self.rms_ped > rms_pedCut))
            image_data = data / 4.
            image_data = image_data - self.mean_ped
            image_data[mySigmaMask] = 0
            flat_image = image_data.flatten()

            counts_i, _ = np.histogram(flat_image, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
            self.countsAll = self.countsAll + counts_i


            supp_coords, supp_weights = al.select_pixels_RMS(image_data, self.rms_ped, self.CLU_CUT_SIGMA)
            zeroSupp_trasposta = supp_coords
            counts2dRaw, _, _ = np.histogram2d(zeroSupp_trasposta[0], zeroSupp_trasposta[1],
                                                               bins=[self.xbins2d, self.ybins2d],
                                                               range=[[0, self.XBINS], [0, self.YBINS]])
            self.countsAll2dRaw = self.countsAll2dRaw + counts2dRaw

            countsZeroSupp_i, _ = np.histogram(supp_weights, bins=2 * self.NBINS, range=(-self.NBINS, self.NBINS))
            self.countsAllZeroSupp = self.countsAllZeroSupp + countsZeroSupp_i

            if self.APPLY_CLUSTERING:
                self.w_clusterAll, self.clu_coordsAll, clu_sizes, clu_baryCoords = clustering_cmos.clustering_v3(
                    np.transpose(supp_coords), supp_weights, myeps=self.myeps)
                cluBary_trasposta = clu_baryCoords.transpose()

                if self.SAVE_EVENTLIST:
                    self.x_allClu = np.append(self.x_allClu, cluBary_trasposta[0])
                    self.y_allClu = np.append(self.y_allClu, cluBary_trasposta[1])
                    self.w_all = np.append(self.w_all, self.w_clusterAll)
                    self.clusizes_all = np.append(self.clusizes_all, clu_sizes)

                counts2dClu, _, _ = np.histogram2d(cluBary_trasposta[0], cluBary_trasposta[1],
                                                             bins=[self.xbins2d, self.ybins2d],
                                                             range=[[0, self.XBINS], [0, self.YBINS]])
                self.countsAll2dClu = self.countsAll2dClu + counts2dClu

                size_mask = np.where(clu_sizes > 0)  # select all clusters!!!!
                countsClu_i, _ = np.histogram(self.w_clusterAll[size_mask], bins=2 * self.NBINS,
                                                   range=(-self.NBINS, self.NBINS))
                self.countsAllClu = self.countsAllClu + countsClu_i

                h_cluSizes_i, _ = np.histogram(clu_sizes, bins=100, range=(0, 100))
                self.h_cluSizeAll = self.h_cluSizeAll + h_cluSizes_i
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
                        data_buffer[6] = np.append(self.w_all, self.w_clusterAll)
                        data_buffer[7] = np.append(self.x_allClu, cluBary_trasposta[0])
                        data_buffer[8] = np.append(self.y_allClu, cluBary_trasposta[1])
                        data_buffer[9] = np.append(self.clusizes_all, clu_sizes)


    def CaptureAnalyze(self):
        camera = checkup()
        lock = Lock() # Data can't be simoultanously analysed by 2 or more processes
        
        # Buffers
        manager = Manager()
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

            temp=[]
            mytime=[]
            k=1
            plt.ion()
            fig3, ax3 = plt.subplots(nrows=1, ncols=2)

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
                        data_queue.put(data)
                        break
                    except Exception as e:
                        camera.stop_video_capture()
                        camera.start_video_capture()
                        print("There are troubles in CaptureAnalyze True:" , e, ", frame = ", i)
                        
                progress_bar_capture.UpdateBar(i)

                # ------------------------------------PLOTS------------------------------------
                if i%10 == 0:
                    self.recover_data(data_buffer)
                    All2dRaw = self.countsAll2dRaw.T
                    #fig3.canvas.flush_events()
                    ax3[0].cla()
                    ax3[0].imshow(np.log10(All2dRaw), interpolation='nearest', origin='lower',
                                  extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
                    ax3[0].set_title("Image Raw", fontsize = 22)
                    ax3[0].set_xlabel("X [pix]", fontsize = 18)
                    ax3[0].set_ylabel("Y [pix]", fontsize = 18)

                    self.bins = np.linspace(-self.NBINS, self.NBINS, 2*self.NBINS+1)
                    self.bins = calP0 + calP1 * self.bins
                    
                    ax3[1].cla()
                    ax3[1].hist(self.bins[:-1], bins=self.bins, weights=self.countsAllZeroSupp, histtype='step', label="pixel thresold", color = "green")
                    ax3[1].hist(self.bins[:-1], bins=self.bins, weights=self.countsAllClu, histtype='step', label='CLUSTERING', color = "red")
                    ax3[1].set_xlabel("Energy [keV]", fontsize = 18)
                    ax3[1].set_ylabel("Counts", fontsize = 18)
                    ax3[1].set_xlim([0,12])
                    ax3[1].set_yscale('log')
                    ax3[1].set_title("Spectrum", fontsize = 22)
                    ax3[1].legend()
                

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
        plt.close(fig3)
        plt.ioff()

        self.recover_data(data_buffer)
        file = open(self.file_path+ 'numero_eventi', 'w')
        file.write(str(self.sample_size))        
        
        # --------------------------------------------------------------------------------------
        # --------------------------------------FINAL PLOT--------------------------------------
        fig6, ax6 = plt.subplots()
        self.countsAll2dClu = self.countsAll2dClu.T
        plt.imshow(self.countsAll2dClu, interpolation='nearest', origin='lower',
                   extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
        plt.colorbar()
        plt.title('hit pixels (rebinned)')

        # plot immagine Raw
        fig3 = plt.figure()
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
        al.write_fitsImage(self.countsAll2dRaw, self.file_path + 'imageRaw' + self.pixMask_suffix + '.fits', overwrite="False")

        print ('SAVE_EVENTLIST=',self.SAVE_EVENTLIST)
        if self.SAVE_EVENTLIST:
            outfileVectors = self.file_path + 'events_list' + self.pixMask_suffix + self.cluCut_suffix + '_v2.npz'
            np.savez(outfileVectors, w=self.w_all, x_pix=self.x_allClu, y_pix=self.y_allClu, sizes=self.clusizes_all)

        plt.show()


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
        
