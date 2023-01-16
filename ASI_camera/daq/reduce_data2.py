import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import subprocess
import time



def red_data(shots_path,ped_file, out_name,cut_threshold=80 ):
    
    #autobkg
    mean_ped=al.read_image(ped_file) # non divido per 4. il ped e' gia' stato diviso alla lettura delle immagini 
    #creo un istogramma vuoto
    x=[]
    countsAll,bins=np.histogram(x,bins=int(65536/4)  ,range=(0,65536/4)  )
    n=0
    n_saved_files=0
    n_letti=0
    supp_weightsAll=np.empty(0)
    x_pix=np.empty(0)
    y_pix=np.empty(0)
    n_img=np.empty(0)
    f=glob.glob(shots_path+"/*.FIT")    
   
    
    for image_file in f:
        n_letti+=1
      #  print('===============>>>  n=',n,'   ',image_file)
       
        try:
            image_data=al.read_image(image_file)/4.
        except:
            print ('file: '+ image_file+ ' not found... skipp!')
            
            continue
        #files_letti.append(image_file)
        # subtract bkg:
        image_data=image_data-mean_ped
        flat_image=image_data.flatten()
        #riempio histo
        counts_i,bins_i=np.histogram(flat_image,bins=int(65536/4)  ,range=(0,65536/4)  )
        countsAll=countsAll+counts_i
        
        supp_coords_i, supp_weigths_i= al.select_pixels2(image_data,cut_threshold)
        traspose=np.transpose(supp_coords_i)
        x_pix=np.append(x_pix,traspose[0])
        y_pix=np.append(y_pix,traspose[1])
        supp_weightsAll=np.append( supp_weightsAll, supp_weigths_i)
        n_img=np.append(n_img,  np.array( [n]*len(x_pix) ) )
        
        if (n_letti%100==0 and n_letti>1) or (n_letti==len(f)):
            n_saved_files+=1
            print('saving '+str(n)+' events, n_file=',str(n_saved_files))
            out_file=shots_path+out_name+'_'+str(n_saved_files)
            al.save_vectors2(out_file, supp_weightsAll,x_pix,y_pix,n_img)
            # AZZERO i VETTORI!!!!!! !!!!!!!!!!!!! @$#@@!!!!!
            x_pix=np.empty(0)
            y_pix=np.empty(0)
            n_img=np.empty(0)
            
        n=n+1

    #end loop files...  
    outHisto_name=shots_path+'histoAll_'+out_name
    al.save_histo(outHisto_name,countsAll,bins)
    cmd='rm '+shots_path+'*.FIT'
    print (cmd)
    subprocess.call(cmd,shell=True)
    print('end loop '+shots_path)








