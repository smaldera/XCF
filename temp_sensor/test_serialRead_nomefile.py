import serial
import time
from datetime import datetime
ser= serial.Serial('/dev/ttyACM0',9600)



firstData=0

Tsum=0.
Hsum=0.
n_sum=0.

def getFileName():
    out_dir='/home/xcf/XCF/temp_data/'
    year=datetime.today().year
    month=datetime.today().month
    filename="temp_sensorData_"+str(year)+"-"+str(month)+".txt"
    return out_dir+filename
    

while 1:
    T=0.
    H=0.
    val = ser.readline()
    val2=val.decode().strip('\n')
   
    val2Splitted=val2.split()
    if val2Splitted[0]=="Found":
        print("sensor found  ... OK")
   
    
    timeStamp=time.time()

    if firstData==0:
         t0=timeStamp
         
        
    #print(val2Splitted)    
    if val2Splitted[0]=="Temp:":
       T=float(val2Splitted[1])
       H=float(val2Splitted[3])
       #print ('time=',time.time()," T=",T, " H= ",H)
       Tsum=Tsum+T
       Hsum=Hsum+H
       n_sum=n_sum+1.
       firstData=1
       if timeStamp-t0>60:
           
           print("===>>>>> Tmedia=",Tsum/n_sum, " Hmedia=",Hsum/n_sum)
           mystring=str(timeStamp)+'  '+str(Tsum/n_sum)+'   '+str(Hsum/n_sum)+'\n'

           outFileName= getFileName()
           with open(outFileName,'a+') as f:
               f.write(mystring)
           Tsum=0.
           Hsum=0.
           n_sum=0.
           t0=timeStamp

