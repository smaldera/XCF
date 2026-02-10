import numpy as np
from matplotlib import pyplot  as plt
from glob import glob
import datetime as dt


filename_list=glob('/home/maldera/IXPE/XCF/data/cmos_temp/temps*.npz')

print(filename_list)

time_all=np.array([])
temp_all=np.array([])

for filename in filename_list:
    data=np.load(filename)

    #print (data['temp'])
    time_all=np.append(time_all,data['time'])
    temp_all=np.append(temp_all,data['temp'])




times_list=[]

for mytime in time_all:
    times_list.append(dt.datetime.fromtimestamp( mytime)) 


plt.plot(times_list  ,temp_all,'ro')
plt.show()
