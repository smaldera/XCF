import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
sys.path.insert(0, '../libs')
import utils as al
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









"""    
shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/test2/CapObj/'
bg_shots_path='/home/maldera/Desktop/eXTP/ASI294/testImages/sensor_4/Fe55/bkg/'

outRootfile_name=shots_path+'histo_all.root'
outHisto_name=shots_path+'histo_all.npz'
cut_threshold=60

pedfile=bg_shots_path+'mean_ped.fits'
mean_ped=al.read_image(pedfile) # non divido per 4. il ped e' gia' stato diviso alla lettura delle immagini 


f=glob.glob(shots_path+"/*/*.FIT")
#creo un istogramma vuoto
x=[]
countsAll,bins=np.histogram(x,bins=int(65536/4)  ,range=(0,65536/4)  )


n=0
n_saved_files=0
f=[]

while 1:
    files_letti=[]
    n_letti=0
    # array vuoti a cui appendo 
    supp_weightsAll=np.empty(0)
    x_pix=np.empty(0)
    y_pix=np.empty(0)
    
    n_loops=0
    while (1):
        f=[]
        f=glob.glob(shots_path+"/*/*.FIT")            
        print ('n loops= ',n_loops,'  len f=',len(f)," type f= ",type(f))
        if len(f)>100:
            break
        n_loops+=1
        time.sleep(1) 
        if n_loops>50:
            print('files testati 50 volte... exit loop, len f=',len(f))
            break
    if (len(f)==0):
        break   #  non ho piu' files, anche dopo il loop... finish! 


#    print(f)
    print('len f=',len(f))
    
    for image_file in f:
        n_letti+=1
        print('===============>>>  n=',n,'   ',image_file)
       
        try:
            image_data=al.read_image(image_file)/4.
        except:
            print ('file: '+ image_file+ ' not found... skipp!')
            
            continue
        files_letti.append(image_file)
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
        
        if (n_letti%100==0 and n_letti>1) or (n_letti==len(f)):
            n_saved_files+=1
            print('saving '+str(n)+' events, n_file=',str(n_saved_files))
            out_file=shots_path+'shots_'+str(n_saved_files)
            al.save_vectors(out_file, supp_weightsAll,x_pix,y_pix)
        
            for l in files_letti:
                cmd=' rm '+l
                print(cmd)
                subprocess.call(cmd,shell=True)
            time.sleep(2)  
           
        n=n+1

al.save_histo(outHisto_name,countsAll,bins)
al.retrive_histo(outHisto_name)   
 
"""
