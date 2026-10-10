import os.path
import os
import argparse
import zwoasi as asi


base_path='/data/cmos_Verticale_new/updateAug2026/'
#base_path='/data/testCMOS/testReadNoise2/1ms_G120/'     
sample_size=10000
bkg_sample_size=1000
EXPO=300000 # us
camera_id=0

### creare loop principale
for i in range(0,5000):
    print("iteration ",i)
    StoreDataIn=base_path+'/'+str(i)+'/'

    cmd='python runOne.py -o '+StoreDataIn+' -s '+str(sample_size)+' -b '+str(bkg_sample_size)+ ' -exp '+str(EXPO) +' -id '+str(camera_id)
    os.system(cmd)
    os.system("cp temps.npz "+StoreDataIn+"temps_"+str(i)+".npz")


print("... all done")


        
