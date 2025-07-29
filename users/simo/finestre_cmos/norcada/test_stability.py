from matplotlib import pyplot as plt
import numpy as np



base_path='/home/maldera/Desktop/eXTP/data/test_finestre/Norcada/scan/'
names=['air_']

fn_time='times.txt'
fn_fit='corrCalib_all.txt'


def read_time(mydir):

    f=open(mydir+'/'+fn_time)
    for l in f:
        tt=l.split()
        t0=float(tt[0])
        t1=float(tt[1])
       
       # print (t0," ",t1)
        return t0+(t1-t0)/2.
        
def read_fit(mydir):

    f=open(mydir+'/'+fn_fit)
    for l in f:
        tt=l.split()

        ampl=float(tt[0].split("=")[1])
        print("ampl=",ampl)
        amplErr=float(tt[1].split("=")[1])
        print("amplErr=",amplErr)
        
        mean=float(tt[2].split("=")[1])
        print("mean=",mean)
        return ampl, amplErr,mean       



def get_values(name):

    time=[]
    ampl_all=[]
    amplErr_all=[]
    
    for i  in range(1,12):
    
        mydir=base_path+name+str(i)+'/'
        #print(i," ",mydir)
        tt=read_time(mydir)
        time.append(tt)
        ampl, amplErr,mean=read_fit(mydir)
        ampl_all.append(ampl)
        amplErr_all.append(amplErr)


    return time, ampl_all, amplErr_all, 
        


timeAir, ampl_allAir, amplErr_allAir= get_values('air_')
timeWin, ampl_allWin, amplErr_allWin= get_values('win_')
  


timeAir=np.array(timeAir)
timeWin=np.array(timeWin)

ampl_allAir=np.array(ampl_allAir)
ampl_allWin=np.array(ampl_allWin)


plt.errorbar(timeAir-timeAir[0], ampl_allAir/np.mean(ampl_allAir), yerr=amplErr_allAir/np.mean(ampl_allAir), fmt="or",label='air'  )
plt.errorbar(timeWin-timeAir[0], ampl_allWin/np.mean(ampl_allWin), yerr=amplErr_allWin/np.mean(ampl_allWin), fmt="ob",label='Norcada')
plt.legend()


plt.figure("fig2")
plt.plot(timeAir-timeAir[0],(ampl_allAir/ampl_allWin)/np.mean(ampl_allAir/ampl_allWin),'or')
#plt.plot(timeAir-timeAir[0],(ampl_allAir)/np.mean(ampl_allAir),'or')


plt.show()
