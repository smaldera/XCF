import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
from pedestal import bg_map

import time
start = time.time()



#hots_list =['/home/maldera/Desktop/eXTP/data/laserShots/nuovoAllineamento/6giu/h_11_m20000/']
bg_shots_path = '/home/maldera/Desktop/eXTP/data/laserShots/nuovoAllineamento/6giu/bkg/'
shots_list = glob.glob('/home/maldera/Desktop/eXTP/data/laserShots/nuovoAllineamento/8giu/alldata')


print(shots_list)



for shots_path in shots_list:

    shots_path=shots_path+'/'
    
    create_bg_map = False
    NBINS=16384  # n.canali ADC (2^14)
    XBINS=2822
    YBINS=4144
    PIX_CUT_SIGMA=15.
    CLU_CUT_SIGMA=150.
    REBINXY=2.
    APPLY_CLUSTERING=True
    SAVE_EVENTLIST=True
    myeps=1.5 # clustering DBSCAN

    xbins2d=int(XBINS/REBINXY)
    ybins2d=int(YBINS/REBINXY)

    
    pixMask_suffix='_pixCut'+str(PIX_CUT_SIGMA)+'sigma'
    cluCut_suffix='_CLUcut_'+str(CLU_CUT_SIGMA)+'sigma' 

    if create_bg_map == True:
        bg_map(bg_shots_path, bg_shots_path + 'mean_ped.fits', bg_shots_path + 'std_ped.fits', draw = 0 )


    # inizio analisi...
    # leggo files pedestal (mean e rms)
    pedfile  = bg_shots_path + 'mean_ped.fits'
    mean_ped = al.read_image(pedfile)
    pedSigmafile  = bg_shots_path + 'std_ped.fits'
    rms_ped = al.read_image(pedSigmafile)


    f = glob.glob(shots_path + "/*.FIT")
    
    # creo istogrammi 1d vuoti
    x = []
    countsAll, bins = np.histogram(x, bins = 2*NBINS, range = (-NBINS,NBINS))
    countsAllZeroSupp, bins = np.histogram(x, bins = 2*NBINS, range = (-NBINS,NBINS))

    countsX_all, binsX_all = np.histogram(x ,  bins =xbins2d , range = (0,XBINS) ) 
    countsY_all, binsY_all = np.histogram(x ,  bins =ybins2d , range = (0,YBINS) ) 

    # creo histo2d vuoto:
    countsAll2dRaw,  xedgesRaw, yedgesRaw= np.histogram2d(x,x,bins=[xbins2d, ybins2d],range=[[0,XBINS],[0,YBINS]])
    
    rms_pedCut=np.mean(rms_ped)+PIX_CUT_SIGMA*np.std(rms_ped)

    print("rms_pedCut=",rms_pedCut)
    # MASCHERA PIXEL RUMOROSI 
    #mySigmaMask=np.where( (rms_ped>10)&(mean_ped>500) )
    mySigmaMask=np.where( (rms_ped>rms_pedCut) )


    #np array vuoti a cui appendo le coordinate
    x_all=np.empty(0)
    y_all=np.empty(0)
    x_allClu=np.empty(0)
    y_allClu=np.empty(0)
    w_all=np.empty(0)
    clusizes_all=np.empty(0)
    n=0.
    # inizio loop sui files
    print('reading files form:',shots_path)
    from tqdm import tqdm
    
    for image_file in tqdm(f, colour='green'):

        n=n+1
        # read image:
        image_data = al.read_image(image_file)/4.
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
        if len( supp_weights)==0:
            print ('vettore vuoto!')
            continue
    
        # salvo pixel che sopravvivono alla selezione:
        #zeroSupp_trasposta= supp_coords.transpose() #!!!!!!
        zeroSupp_trasposta= supp_coords

        #x_all=np.append(x_all,zeroSupp_trasposta[0])
        #y_all=np.append(y_all,zeroSupp_trasposta[1])

        #istogramma 2d immagine raw dopo zero suppression:
        counts2dRaw,  xedgesRaw, yedgesRaw= np.histogram2d(zeroSupp_trasposta[0],zeroSupp_trasposta[1],weights=supp_weights, bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
        countsAll2dRaw=countsAll2dRaw+counts2dRaw

        #spettro dopo zeroSuppression
        countsZeroSupp_i, bins_i = np.histogram( supp_weights,  bins = 2*NBINS, range = (-NBINS,NBINS) ) 
        countsAllZeroSupp =countsAllZeroSupp  +  countsZeroSupp_i


        # X projection:
        countsX_i, binsX_i = np.histogram(zeroSupp_trasposta[0] ,  weights=supp_weights,  bins =xbins2d , range = (0,XBINS) ) 
        countsX_all=countsX_all+countsX_i
        #Y projection:
        countsY_i, binsY_i = np.histogram(zeroSupp_trasposta[1] , weights=supp_weights, bins =ybins2d , range = (0,YBINS) ) 
        countsY_all=countsY_all+countsY_i
    

        #if SAVE_EVENTLIST:
        #        x_allClu=np.append(x_allClu,cluBary_trasposta[0])
        #        y_allClu=np.append(y_allClu,cluBary_trasposta[1])
        #        w_all=np.append(w_all,w_clusterAll)
        #        clusizes_all=np.append(clusizes_all,clu_sizes)
    


        ###########
        # plot immagine


    # plot immagine Raw
    fig3, ax3 = plt.subplots()
    countsAll2dRaw=   countsAll2dRaw.T
    plt.imshow((countsAll2dRaw), interpolation='nearest', origin='lower',  extent=[xedgesRaw[0], xedgesRaw[-1], yedgesRaw[0], yedgesRaw[-1]]   ) 
    plt.colorbar()
    plt.title('pixels>zero_suppression threshold')


    print("xedges = ",xedgesRaw[0],"  ", xedgesRaw[-1])
    # plot spettro
    fig, h1 = plt.subplots()

    h1.hist(bins[:-1], bins = bins, weights = countsAllZeroSupp, histtype = 'step',label="pixel thresold")
    plt.legend()
    plt.title('spectra')



    #plot X projection:
    fig2, h3 = plt.subplots()
    #fig2=plt.figure(2)
    h3.hist(binsX_all[:-1], bins = binsX_all, weights =   countsX_all, histtype = 'step',label="X projection")



    fig3, h4 = plt.subplots()
    h4.hist(binsY_all[:-1], bins = binsY_all, weights =   countsY_all, histtype = 'step',label="Y projection")
    plt.legend()


    # save histos
    np.savez(shots_path+'spectrum_all_ZeroSupp'+pixMask_suffix+cluCut_suffix, counts = countsAllZeroSupp, bins = bins)
    np.savez(shots_path+'X_dist_ZeroSupp'+pixMask_suffix+cluCut_suffix, counts = countsX_all, bins = binsX_all)
    np.savez(shots_path+'Y_dist_ZeroSupp'+pixMask_suffix+cluCut_suffix, counts = countsY_all, bins = binsY_all)



    # save figures
    al.write_fitsImage(countsAll2dRaw, shots_path+'imageRaw'+pixMask_suffix +'.fits'  , overwrite = "False")

    # salva vettori con event_list:
    #if SAVE_EVENTLIST:
    #  outfileVectors= shots_path+'events_list'+pixMask_suffix+cluCut_suffix+'_v2.npz'
    #  print('writing events in:',outfileVectors)
    #al.save_vectors(outfileVectors, w_all, x_allClu, y_allClu,clusizes_all)
    #  np.savez(outfileVectors, w=w_all, x_pix=x_allClu, y_pix=y_allClu, sizes=clusizes_all)
    plt.show()


end = time.time()
print(end - start) 


