
from astropy.io import fits as pf
from matplotlib import pyplot as plt

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics

#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-09_13_41_13Z/2021-12-09-1341_2-CapObj_0000.FIT'
file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-13_11_52_30Z/2021-12-13-1152_5-CapObj_0000.FIT'
#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-13_13_16_21Z/2021-12-13-1316_3-CapObj_0000.FIT'   # tutto saturo, 16 bit
#file_path='/home/maldera/Desktop/eXTP/ASI294/testImages/CapObj/2021-12-13_13_17_20Z/'  

data_f = pf.open(file_path, memmap=True)
data_f.info()

image_data = pf.getdata(file_path, ext=0)/4.
print(image_data.shape[1])

print(image_data)
coord_arr=[]
weights=[]

for i in range(0,image_data.shape[0]):
      for j in range (0,image_data.shape[1] ):
          coord_arr.append([i,j])
          weights.append(image_data[i,j])

#print(coord_arr)
          
coord2=np.array(coord_arr)
weights2=np.array(weights)

#print(coord2[0])

mask_zeroSupp=np.where(weights2>330)
supp_coords=coord2[mask_zeroSupp]
supp_weights=weights2[mask_zeroSupp]




print('START CLUSTERING...')

db = DBSCAN(eps=0.5, min_samples=1, n_jobs=1, algorithm='ball_tree').fit(supp_coords)
#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

#print("labels=",type(labels))

unique_labels=set(labels) # il set elimina tutte le ripetizioni

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


for clu_id in unique_labels:
      
      print ("CLUSTER_ID=",clu_id)
      if clu_id==-1:
            continue
      
      clu_mask=np.where(labels==clu_id)
      clu_coords=supp_coords[clu_mask]
      clu_weights=supp_weights[clu_mask]

      print("cluster coord=",clu_coords)
      print("cluster weights=",clu_weights)
      






plt.figure()
plt.hist(supp_weights, bins=16384, range=(0,16384)   , alpha=1, histtype='step')
# istogramma 2d

traspose=np.transpose(supp_coords)
x=traspose[0]
y=traspose[1]
plt.figure()
plt.hist2d(x,y, bins=[image_data.shape[0],image_data.shape[1] ],  weights=supp_weights)
plt.colorbar()
#plt.imshow(image_data, cmap='plasma')
plt.show()
