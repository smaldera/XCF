import numpy as np
from PiMove import PiMikro
import time
import ConnectToWSL
import sys
sys.path.insert(0, "C:\Users\XCF\Desktop\Git_CMOS\XCF-cmosLargeEvents\ASI_camera\GUI")
import os
import zwoasi as asi
from gui_analyzer_parallelSaveLargeClu import aotr2


def Connect2CMOS(bkg_path, data_path):
    WB_R=50
    WB_B=50
    EXPO=300000 # us
    GAIN=120
    CAMERA_ID=0
    print ("out path= ",base_path)
    print ("sample_size= ",sample_size)
    print ("bkg size= ",bkg_sample_size)
    print ("Exposure= ",EXPO)
    print ("camera_id= ",CAMERA_ID)
    xyRebin=5
    sigma=10
    cluster=10
    ApplyClustering=True
    SaveEventList=True
    raw=False
    Eps=1.5
    num_jobs=4   # num paraller jobs
    leng=500
    save_every=10

    env_filename = os.getenv('ZWO_ASI_LIB')
    asi.init(env_filename)
    StoreDataIn=data_path

    try:
        print("inizialising acq loop:")
        OBJ = aotr2(StoreDataIn, sample_size, WB_R, WB_B, EXPO, GAIN, bkg_path, xyRebin, sigma, cluster, ApplyClustering, SaveEventList, raw, Eps,num_jobs , leng,save_every, camera_id=CAMERA_ID, LIVE_PLOTS=False, GUI=False)
        print("starting acq loop:")
        OBJ.CaptureAnalyze() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    except Exception as e:
        print("\n There are troubles in CaptureAndAnalyze: ", e)

    finally:
        print ("del OBJ")
        del OBJ


Tempo = '120' #NOTE: livetime MUST be passed like a string value
path = 'Desktop/SaveNpz/19_02_25/' #NOTE: this path needs to be relative to 'C/Users/XCF/'. So if you's like to save in Desktop, type 'Desktop/'
loops = 2

Pi = PiMikro()
Pi.MoveThat('ylow', 5., velocity=3)
Pi.MoveThat('xlow', 232., velocity=3)
Pi.MoveThat('xup', 12.5, velocity=3)
Pi.MoveThat('yup', 12.5, velocity=3)

xA = [300., 240]
labels = ['Air', 'NORCADA']
print('Moving to home position: ', labels[0])
Pi.MoveThat('xlow', xA[0], velocity=3)

for j in range(1, loops+1):
    i = 0
    for pos in xA:
        print('Moving to ', labels[i])
        Pi.MoveThat('xlow', pos, velocity=3)
        time.sleep(2)
        #ConnectToWSL.callps1(path=path, livetime=Tempo, name=labels[i]+'_10kV_0.006mA')
        #print('Saving spectrum')
        i+=1
