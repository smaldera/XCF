from astropy.io import fits as pf
import numpy as np
from matplotlib import pyplot as plt



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


def write_fitsImage(array, nomefile):
   hdu = pf.PrimaryHDU(array)
   hdu.writeto(nomefile)



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


def select_pixels2(image_data, threshold=100): # much better!!


   mask_zeroSupp=np.where(image_data>threshold)
   #debug:
   # print("mask=",mask_zeroSupp)
   # print("maksed array=",image_data[mask_zeroSupp])
   
   supp_coords=np.transpose(mask_zeroSupp)
   supp_weights=image_data[mask_zeroSupp]

   
   
   return supp_coords, supp_weights

   
def clustering(supp_coords,supp_weights ):

   print('START CLUSTERING...')

   db = DBSCAN(eps=2, min_samples=2, n_jobs=1, algorithm='ball_tree').fit(supp_coords)
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
      if len(clu_weights)<=3:
         #sum_w.append(clu_weights.sum() ) 
         sum_w.append(np.sum(clu_weights) ) 
        
   return sum_w