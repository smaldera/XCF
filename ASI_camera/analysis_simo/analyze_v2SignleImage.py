import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
from pedestal import bg_map
#import clustering_cmos 
from sklearn.cluster import DBSCAN
import time
start = time.time()




shots_path = '/home/maldera/Desktop/eXTP/data/ASI_testMu/testCMOS_verticale/data2/'
bg_shots_path ='/home/maldera/Desktop/eXTP/data/ASI_testMu/testCMOS_verticale/data2/bkg/'

calP0=-0.003201340833319255
calP1=0.003213272145961988


create_bg_map = False
NBINS=16384  # n.canali ADC (2^14)
XBINS=2822
YBINS=4144
PIX_CUT_SIGMA=15.
CLU_CUT_SIGMA=15.
REBINXY=1.
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


f = glob.glob(shots_path + "/img_cluSize*.fits")

# creo istogrammi 1d vuoti
x = []
countsAll, bins = np.histogram(x, bins = 2*NBINS, range = (-NBINS,NBINS))
countsAllZeroSupp, bins = np.histogram(x, bins = 2*NBINS, range = (-NBINS,NBINS))
countsAllClu, bins = np.histogram(x,  bins = 2*NBINS, range = (-NBINS,NBINS))
h_cluSizeAll,binsSize=np.histogram(x,bins=100, range=(0,100))

# creo histo2d vuoto:
countsAll2dClu,  xedges, yedges=       np.histogram2d(x,x,bins=[xbins2d, ybins2d],range=[[0,XBINS],[0,YBINS]])
countsAll2dRaw,  xedgesRaw, yedgesRaw= np.histogram2d(x,x,bins=[xbins2d, ybins2d],range=[[0,XBINS],[0,YBINS]])

#zero_img = np.zeros((XBINS, YBINS))
#image_SW = np.zeros((XBINS, YBINS))

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

#for i in tqdm (range (len(f)), desc="Loading..."):
for image_file in tqdm(f, colour='green'):
    print(n," --> ", image_file)
  #  if n%10==0:
  #       frac=float(n/len(f))*100.
  #       print(" processed ",n," files  (  %.2f %%)" %frac )

    n=n+1

    # read image:
    image_data = al.read_image(image_file)/4.

    print('image_data.sahpe=', image_data.shape)
    
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
    #supp_coords, supp_weights=al.select_pixels2(image_data, 150)
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
    counts2dRaw,  xedgesRaw, yedgesRaw= np.histogram2d(zeroSupp_trasposta[0],zeroSupp_trasposta[1],weights=supp_weights,bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
    countsAll2dRaw=countsAll2dRaw+counts2dRaw

    #spettro dopo zeroSuppression
    countsZeroSupp_i, bins_i = np.histogram( supp_weights,  bins = 2*NBINS, range = (-NBINS,NBINS) ) 
    countsAllZeroSupp =countsAllZeroSupp  +  countsZeroSupp_i
    #CLUSTERING
    supp_coords= np.transpose(supp_coords)
    if APPLY_CLUSTERING:
       
        # test clustering.... # uso v2 per avere anche le posizioni
        # w_clusterAll, clu_coordsAll, clu_sizes, clu_baryCoords    =clustering_cmos.clustering_v3(np.transpose(supp_coords),supp_weights,myeps=myeps) 
        # cluBary_trasposta= clu_baryCoords.transpose()
        # RIFACCIO QUA IL CLUSTERING x AVERE TUTTE LE INFO:\
        coordsAll=np.empty((0,0))
        db = DBSCAN(eps=1.5, min_samples=1, n_jobs=1, algorithm='ball_tree').fit(supp_coords)     
        labels = db.labels_

        #print("labels=",type(labels))
        unique_labels=set(labels) # il set elimina tutte le ripetizioni
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        sum_w=[]   
        for clu_id in unique_labels:
      
            print ("CLUSTER_ID=",clu_id)
            if clu_id==-1:
                continue
      
            clu_mask=np.where(labels==clu_id)
            clu_coords=supp_coords[clu_mask]
            #coordsAll=np.append(coordsAll, clu_coords)
            clu_weights=supp_weights[clu_mask]

            # apply enegy caliration from x-rays: ?????????????????????????????????
            clu_weights=clu_weights*calP1+calP0
            print ("clu_coords=",clu_coords)
            print ("clu_weight=",clu_weights)
            clu_size=len(clu_weights)
            print("clu_size=",clu_size )
            if clu_size>9:
                #plt.plot( clu_coords.T[0], clu_coords.T[1],'o')
                counts2dClu,  xedges2dClu, yedges2dClu= np.histogram2d(clu_coords.T[0], clu_coords.T[1],weights=clu_weights,bins=[xbins2d, ybins2d ],range=[[0,XBINS],[0,YBINS]])
                counts2dClu=   counts2dClu.T
                plt.imshow(counts2dClu, interpolation='nearest', origin='lower',  extent=[xedges2dClu[0], xedges2dClu[-1], yedges2dClu[0], yedges2dClu[-1]])
                plt.colorbar()
                plt.xlim(min(clu_coords.T[0])-10,max(clu_coords.T[0])+10)
                plt.ylim(min(clu_coords.T[1])-10,max(clu_coords.T[1])+10)
                             
                
                plt.show()
            # qua disegno la traccia!!!!

        
       
        
       

###########
# plot immagine
fig2, ax2 = plt.subplots()



