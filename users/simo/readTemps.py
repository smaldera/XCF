
import numpy as np
import random
import datetime as dt
import glob
from matplotlib import pyplot as plt



def create_plots(dir_path):

    fList = glob.glob(dir_path + "/temp*.txt")
    print(fList)
    time=[]
    temp=[]
    hum=[]
    
    for filename in fList:
        f=open(filename)
    
        for line in f:
         #  print(line[:-1].split())

           time.append(dt.datetime.fromtimestamp(  (float(line[:-1].split()[0]))   ))
           temp.append(float(line[:-1].split()[1]))
           hum.append(float(line[:-1].split()[2])  )
           #print('time=',time," temp= ",temp,' humidity= ',hum)

    return time,temp,hum   






if __name__ == '__main__':
   
    dir_path='/home/maldera/Desktop/eXTP/data/temp_data/'
    time,temp,hum  = create_plots(dir_path)

   
    plt.plot(time,temp,'pb')
    plt.show()
   
    
