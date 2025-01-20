import numpy as np
from sklearn.cluster import DBSCAN


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

#   print('START CLUSTERING...')
   
   coordsAll=np.empty((0,0))
   db = DBSCAN(eps=1.5, min_samples=1, n_jobs=1, algorithm='ball_tree').fit(supp_coords)
  
   #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
   #core_samples_mask[db.core_sample_indices_] = True
   labels = db.labels_

   #print("labels=",type(labels))
   unique_labels=set(labels) # il set elimina tutte le ripetizioni
   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
   n_noise_ = list(labels).count(-1)
#   print("Estimated number of clusters: %d" % n_clusters_)
#   print("Estimated number of noise points: %d" % n_noise_)

   sum_w=[]   
   for clu_id in unique_labels:
      
    #  print ("CLUSTER_ID=",clu_id)
      if clu_id==-1:
            continue
      
      clu_mask=np.where(labels==clu_id)
      clu_coords=supp_coords[clu_mask]
      coordsAll=np.append(coordsAll, clu_coords)
      
      clu_weights=supp_weights[clu_mask]

     # print("type cluster coord=",clu_coords.shape)
    #  print("cluster coordAll=",coordsAll)
     
    #  print("cluster weights=",clu_weights, "sum= ",clu_weights.sum())
     # if len(clu_weights)==1:
     #    sum_w.append(clu_weights.sum() ) 
      sum_w.append(np.sum(clu_weights) ) 

      # DEBUG
      #if clu_id>10:  #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      #   break
     
    #  print("cluster coordAll=",coordsAll)
    #  print("cluster coordAll reshape= ", coordsAll.reshape(int(len(coordsAll)/2),2 ))
      
      
   return sum_w,coordsAll.reshape(int(len(coordsAll)/2),2 )




def clustering_v3(supp_coords,supp_weights,myeps=1):

#   print('START CLUSTERING...')
   
   coordsAll=np.empty((0,0))
   cg_coords = np.empty((0,0))
   sum_w=[]
   clu_size=[]

   print (" len(supp_coords=", len(supp_coords))
   if len(supp_coords)==0:
      print("clustering: vettore vuoto... skip event")
      return(np.array(sum_w),coordsAll.reshape(int(len(coordsAll)/2),2 ), np.array(clu_size),  cg_coords.reshape(int(len(cg_coords)/2),2))
   
   db = DBSCAN(eps=myeps, min_samples=1, n_jobs=1, algorithm='ball_tree').fit(supp_coords)
  
  
   labels = db.labels_
   unique_labels=set(labels) # il set elimina tutte le ripetizioni
   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
   n_noise_ = list(labels).count(-1)
#   print("Estimated number of clusters: %d" % n_clusters_)
#   print("Estimated number of noise points: %d" % n_noise_)

   
   for clu_id in unique_labels:
      
      # print ("CLUSTER_ID=",clu_id)
      if clu_id==-1:
            continue
      
      clu_mask=np.where(labels==clu_id)
      clu_coords=supp_coords[clu_mask]
      coordsAll=np.append(coordsAll, clu_coords)
      clu_weights=supp_weights[clu_mask]
      # print("cluster weights=",clu_weights, "sum= ",clu_weights.sum())

      #### baricentro cluster:
      x_cg = 0
      y_cg = 0
      t = clu_coords.transpose()
      i=0
      while(i < len(clu_weights)):
           
            x_cg = x_cg + t[0][i] * clu_weights[i]
            y_cg = y_cg + t[1][i] * clu_weights[i]
            i = i + 1
            #print("x_cg = ", x_cg, " y_cg = ", y_cg)

      cg_coords = np.append(cg_coords, np.array([x_cg, y_cg] / np.sum(clu_weights)))
      ######## end baricentro cluster
            
      sum_w.append(np.sum(clu_weights) )
      clu_size.append(len(clu_weights))

      # DEBUG
      #if clu_id>10:  #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      #   break
     
          
      
   return np.array(sum_w),coordsAll.reshape(int(len(coordsAll)/2),2 ), np.array(clu_size),  cg_coords.reshape(int(len(cg_coords)/2),2)
   







