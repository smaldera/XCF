
import os



base_path='/home/maldera/Desktop/eXTP/data/test_finestre/Norcada/scan/'

#names=['win_','air_']
names=['air_']

for name in names:

    for i  in range(1,12):
    
        mydir=base_path+name+str(i)+'/'
        print(i," ",mydir)
        
        cmd='ls '+mydir+'events_list_pixCut10sigma_CLUcut_10sigma_v2.npz >'+mydir+'/file_list.txt' 
        print(cmd)
        os.system(cmd)

        cmd= 'python read_eventLists_spots.py -in '+mydir+'file_list.txt -dir ' + mydir
        print("CMD=",cmd)
        try:
            os.system(cmd)
        except:
            print("errr..")
            break
