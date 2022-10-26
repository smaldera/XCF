from astropy.io import fits as pf
import numpy as np
from matplotlib import pyplot as plt
from scipy import asarray as ar,exp


from sklearn.cluster import DBSCAN
from sklearn import metrics

def read_image(nomefile):
   data_f = pf.open(nomefile, memmap=True)
   #data_f.info()
   image_data = pf.getdata(nomefile, ext=0) # NON dividi per 4!!!!

   return image_data


def plot_image(image_data):
    plt.figure()
    plt.imshow(image_data, cmap='plasma')
    plt.colorbar()
    plt.show()


def isto_all(image_data):
    flat_image=image_data.flatten()
    fig, ax = plt.subplots()
    ax.hist(flat_image, bins=int(65536/4), range=(0,65536/4)   , alpha=1, histtype='step')
    mean=flat_image.mean()
    rms=flat_image.std()
    s='mean='+str(round(mean,3))+"\n"+"RMS="+str(round(rms,3))
    ax.text(0.7, 0.9, s,  transform=ax.transAxes,  bbox=dict(alpha=0.7))
    plt.show()

def isto_all_root(image_data):
   #h2=ROOT.TH1F('h2','',16384,0,16384)
   image_data=np.float64(image_data) #!!!!!! okkio, senza questo fillN a volte non va
   flat_image=image_data.flatten()
   h2=ROOT.TH1F('h2','',16384,0,16384)
   w=np.ones(len(flat_image))
   print(w)
   h2.FillN(len(flat_image), flat_image, w)
   return h2
  
def write_fitsImage(array, nomefile,overwrite='False' ):
   hdu = pf.PrimaryHDU(array)
   hdu.writeto(nomefile,overwrite=overwrite)



def select_pixels(image_data):  # DEPRECATED
   coord_arr=[]
   weights=[]

   for i in range(0,image_data.shape[0]):
      for j in range (0,image_data.shape[1] ):
          coord_arr.append([i,j])
          weights.append(image_data[i,j])


          
   coord2=np.array(coord_arr)
   weights2=np.array(weights)

   #print(coord2[0])

   mask_zeroSupp=np.where(weights2>100)
   supp_coords=coord2[mask_zeroSupp]
   supp_weights=weights2[mask_zeroSupp]

   return supp_coords, supp_weights


def select_pixels2(image_data, threshold = 100, upper = 100000): # much better!!


   mask_zeroSupp=np.where( (image_data>threshold) & (image_data<upper))
   #debug:
   # print("mask=",mask_zeroSupp)
   # print("maksed array=",image_data[mask_zeroSupp])
   
   supp_coords = np.transpose(mask_zeroSupp)
   supp_weights = image_data[mask_zeroSupp]

      
   return supp_coords, supp_weights

   
def clustering(supp_coords,supp_weights ):

   print('START CLUSTERING...')
   
   
   db = DBSCAN(eps=1, min_samples=2, n_jobs=1, algorithm='ball_tree').fit(supp_coords)
   #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
   #core_samples_mask[db.core_sample_indices_] = True
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
      clu_weights=supp_weights[clu_mask]

      print("cluster coord=",clu_coords)
      print("cluster weights=",clu_weights, "sum= ",clu_weights.sum())
      #if len(clu_weights)<=3:
         #sum_w.append(clu_weights.sum() ) 
      sum_w.append(np.sum(clu_weights) ) 
        
   return sum_w

def clustering_v2(supp_coords,supp_weights):

    #print("\n")
    #print('START CLUSTERING...')
    coordsAll = np.empty((0,0))
    
    clu_lenghts = np.empty((0,0))   #matrice che contiene tutte le lunghezze deu cluster
    
    cg_coords = np.empty((0,0))
    
    #db = DBSCAN(eps=1, min_samples=2, n_jobs=1, algorithm='ball_tree').fit(supp_coords)
    db = DBSCAN(eps = 1, min_samples = 1, n_jobs = 1, algorithm = 'ball_tree').fit(supp_coords)
  
    #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    #print("labels=",type(labels))

    unique_labels = set(labels) # il set elimina tutte le ripetizioni

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    #print("\n")
    #print("Estimated number of clusters: %d" % n_clusters_)
    #print("Estimated number of noise points: %d" % n_noise_)
    #print("\n")
    
    sum_w = []
    i = 0
    for clu_id in unique_labels:
      
    #  print ("CLUSTER_ID=",clu_id)
        if clu_id == -1:    #cluster che sono rumore
            continue
      
        clu_mask = np.where(labels == clu_id)
        clu_coords = supp_coords[clu_mask]
        #print("supp_coords = ", supp_coords[clu_mask])
        clu_lenghts = np.append(clu_lenghts, len(clu_coords))   #contiene le dimensioni dei cluster
        
            
        coordsAll = np.append(coordsAll, clu_coords)
        
        clu_weights = supp_weights[clu_mask]
        
        
        
        t = clu_coords.transpose()
        
        #print("t[0] = ", t[0], " t[1] = ", t[1])
        #print("clu_lenghts = ", clu_lenghts[clu_id])
        #print("clu_weights = ", clu_weights)
        
        i = 0
        x_cg = 0
        y_cg = 0
        
        while(i < len(clu_coords)):
        
            x_cg = x_cg + t[0][i] * clu_weights[i]
            y_cg = y_cg + t[1][i] * clu_weights[i]
            
            i = i + 1
            
        #print("x_cg = ", x_cg, " y_cg = ", y_cg)
            
        sum_w.append(np.sum(clu_weights))
            
        cg_coords = np.append(cg_coords, np.array([x_cg, y_cg] / np.sum(clu_weights)))
        
        
        #print("cg_coords = ", cg_coords)
        #print("cluster coords = ", clu_coords)
        #print("cluster weights = ", clu_weights)
        #print("\n\n\n")
      #if len(clu_weights) <= 3:
         #sum_w.append(clu_weights.sum() ) 
        

      # DEBUG
        #if clu_id > 10:  #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #break
     
    #  print("cluster coordAll=",coordsAll)
    #  print("cluster coordAll reshape= ", coordsAll.reshape(int(len(coordsAll)/2),2 ))
      
    #print("max lenght = ", np.max(clu_lenghts))  #mi serve l'indice per cui ho max lenght cosí individuo le coordinate del cluster piu grande e plotto solo quello
    #print("min lenght = ", np.min(clu_lenghts))
    #print("\n")
        
    #print(coordsAll.reshape(int(len(coordsAll)/2),2)[i])
    
    return sum_w, coordsAll.reshape(int(len(coordsAll)/2),2), clu_lenghts, cg_coords.reshape(int(len(cg_coords)/2),2)



def how_it_works(coordsAll, clu_lenghts, supp_coords):   #plotta scatter plot che mostra soltanto il cluster più grande

    x_all = np.empty(0)
    y_all = np.empty(0)
    x_allClu = np.empty(0)
    y_allClu = np.empty(0)

    start = 0   #start e delta indicano rispettivamente l'indice di inizio del cluster e la lunghezza del cluster da sommare a start per trovare l'indice della fine del cluster
    delta = np.max(clu_lenghts)
    
    i = 0   #indice del ciclo
    
    while(clu_lenghts[i] < np.max(clu_lenghts)):
        start = int(start + clu_lenghts[i])
        i = i + 1
        
    end = int(start + np.max(clu_lenghts))
    #print(end)
    
    i  = start
    
    coordsExample = np.empty((0,0)) #matrice che contiene solo le coordinate del cluster d'esempio
    
    while(i < end):
        #print(coordsAll[i])     #print per controllare che effettivamente le coordinate trovate corrispondano al primo cluster trovato di coordinate massime
        coordsExample = np.append(coordsExample, coordsAll[i])
        #print(coordsExample)
        i = i + 1
        
    coordsExample = coordsExample.reshape(int(len(coordsExample)/2),2)
    #print("coordsExample ", coordsExample)
    
    trasposta = supp_coords.transpose()
    x_all = np.append(x_all, trasposta[0])
    y_all = np.append(y_all, trasposta[1])

    clu_trasposta = coordsExample.transpose()
    x_allClu = np.append(x_allClu, clu_trasposta[0])
    y_allClu = np.append(y_allClu, clu_trasposta[1])
    
    fig, ax = plt.subplots()
    plt.plot(x_all, y_all, 'sr', alpha = 0.3, ms = 10)
    plt.plot(x_allClu, y_allClu, 'sg', alpha = 1, markerfacecolor = 'none', ms = 11)
    
    return


#def centre_of_gravity(coordsAll, clu_lenghts, clu_weights):

    




# function to retrive saved numpy arrayis...

def save_vectors(out_file, supp_weightsAll,x_pix,y_pix):
    np.savez(out_file,w=supp_weightsAll, x_pix=x_pix, y_pix=y_pix)


def save_vectors2(out_file, supp_weightsAll,x_pix,y_pix,n_img):
       np.savez(out_file,w=supp_weightsAll, x_pix=x_pix, y_pix=y_pix,n_img=n_img)
       
    
def save_histo(outHisto_name,countsAll,bins):
    np.savez(outHisto_name,counts=countsAll,bins=bins)
     
def retrive_vectors(nomefile):
    data=np.load(nomefile)
    w=data['w']
    x_pix=data['x_pix']
    y_pix=data['y_pix']

    return w,x_pix,y_pix



def retrive_histo(nomefile):
    data=np.load(nomefile)
    counts=data['counts']
    bins=data['bins']
    fig, ax = plt.subplots()
    #print ("len(coutsAll)=",len(countsAll) )
    histo=ax.hist(bins[:-1],bins=bins,weights=counts, histtype='step')
    #plt.show()
    return histo
    
    
def gaus(x, a, x0, sigma):
    return a * exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
def retta(x, a, b):
    return (a * x) + b
    
def linear_func(p, x):
   m, c = p
   return m * x + c
   
def err_retta(x, a, b, erra, errb):

    return np.sqrt((x ** 2 * erra ** 2) + (errb ** 2))
