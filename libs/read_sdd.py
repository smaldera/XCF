import numpy as np
from matplotlib import pyplot as plt
from astropy.time import Time


def pharse_mca(filename):
   
   f=open(filename,encoding="utf8", errors='ignore')
   i=0
   read_data=False
   #w=np.zeros(n_bins)
   deadTime=''
   livetime=''
   fast_counts=''
   data=[]
   for line in f:
      
       #print (line[:-1])


       if line[:-1]=="<<END>>":
           #print(line[:-1])
           read_data=False
           continue
       
       if line[:-1]=="<<DATA>>":
           #print(line[:-1])
           read_data=True
           continue

       
           
       if (read_data): 
          data.append(int(line[:-1]))
          continue
        
       if line.split()[0]=='Dead':
            try:
                print ("dead=",line.split()[2])
                deadTime= line.split()[2]
            except:
                deadTime='' 
            continue    
             
       if line.split()[0]=='Fast' and   line.split()[1]=='Count:' :
            try:
                print ('counts=',line.split()[2])
                fast_counts= float(line.split()[2])
            except:
                fast_counts='' 
            continue   
                
       if line.split()[0]=='LIVE_TIME':
            try:
                print ('live=',line.split()[2])
                livetime= float(line.split()[2])
            except:
                livetime=''   
            continue   
       
       if line.split()[0]=='START_TIME':
            try:
                print ('start=',line.split()[2],' ',line.split()[3])
                hour = line.split()[3]
                date = line.split()[2]
                seconds = float(hour.split(':')[2])+60*float(hour.split(':')[1])+3600*float(hour.split(':')[0])
                month = date.split('/')[0]
                day = date.split('/')[1]
                year = date.split('/')[2]
                date_iso = year+'-'+month+'-'+day+'T'+hour
                MJD = Time(date_iso, format='isot').mjd
                start = MJD

            except:
                start=''   
            continue     
        
        

   data_array=np.array(data)               
                
   return data_array, deadTime, livetime, fast_counts, start





if __name__ == "__main__":

    mca_file='/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/SDD/Fe_14Oct2022_5mm.mca'
   # mca_file= '/home/maldera/Desktop/eXTP/ASI294/testImages/eureca_noVetro/misure_collimatore_14Oct/SDD/Fe_14Oct2022_2mm.mca'
    data_array, deadTime, livetime, fast_counts, start =pharse_mca(mca_file)
    print("livetime=",livetime,"counts=", fast_counts, "RATE=",fast_counts/livetime,' Hz' )
    print("deadTime=",deadTime,"starting=",start)



    #plot
    size=len(data_array)      
    bin_edges=np.linspace(0,size+1,size+1)

    fig1=plt.figure(1)
    plt.hist(bin_edges[:-1],bins=bin_edges,weights=data_array, histtype='step')
    plt.show()




