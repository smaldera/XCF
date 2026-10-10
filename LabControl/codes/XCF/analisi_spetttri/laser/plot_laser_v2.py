
import numpy as np
from  matplotlib import pyplot as plt


def distance_mm(x0,y0,x1,y1):

    d= ((x0-x1)**2 + (y0-y1)**2)**0.5

    d=d*4.6*1e-3
    return d
    

def apply_rot(x,y,theta):

    x1=x*np.cos(theta)-y*np.sin(theta)
    y1=x*np.sin(theta)+y*np.cos(theta)


      
    return (x1,y1)
      


def read_spotfile(nomefile):

    
    mydict={}
    myfile=open(nomefile)
    i=0
    for l in myfile:
        splitted=l[:-1].split()
        print(splitted)
        title=(splitted[0])
        x=float(splitted[1])
        x_err=float(splitted[2])
        y=float(splitted[3])
        y_err=float(splitted[4])

        if i==0:
            x0=x
            y0=y
        i=i+1 
            
        mydict[title]=[x,x_err,y,y_err]

   
    return mydict

        


#nomefile='/home/maldera/Desktop/eXTP/data/laserShots/nuovoAllineamento/8giu/plots/out.txt'
#keys=['h_7.65+20k-20k', 'h_7.65_m10k', 'h_7.65_centro_dopo-10k+10k','h_7.65_p10k', 'h_7.65_centroFinale']


nomefile='/home/maldera/Desktop/eXTP/data/laserShots/nuovoAllineamento/14giu/plots/out.txt'
keys=['h_7.65_0_0', 'h_7.65_p10k', 'h_7.65_p10k_m10k','h_7.65_p10k_m20k', 'h_7.65_p20k_m20k']






d=read_spotfile(nomefile)      

x_all=[]
y_all=[]

theta=-np.arctan(0.021)
#theta=0.

for k in keys:

    x=d[k][0]
    y=d[k][2]
    #APPLY ROTATION!!!
    rot=apply_rot(x,y,theta)
    x=rot[0]
    y=rot[1] 
    d[k][0]=x
    d[k][2]=y
    x_all.append(x)
    y_all.append(y)
  

print(d)


# APPLY ROTATION




fig=plt.figure(1)

plt.plot(d[keys[0]][0], d[keys[0]][2] ,'o', label='starting center')
plt.plot(d[keys[1]][0], d[keys[1]][2] ,'o', label='p10k')
plt.plot(d[keys[2]][0], d[keys[2]][2] ,'o', label='new center +10k  -10k')
plt.plot(d[keys[3]][0], d[keys[3]][2] ,'o', label='-10k')
plt.plot(d[keys[4]][0], d[keys[4]][2] ,'o', label='center +20k  -20k')

plt.plot(x_all,y_all,'--')

print('d[h_7.65_0_0]= ',d[keys[0]])
print('d[h_7.65_p10k]= ',d[keys[1]])
print('d[h_7.65_p10k_m10k]= ',d[keys[2]])
print('d[h_7.65_p10k_m20k]= ',d[keys[3]])
print('d[h_7.65_p20k_m20k]= ',d[keys[4]])



dp1=abs(d[keys[0] ][0]-d[keys[1]][0])
#dp1=( (d['h_7.65_0_0'][0]-d['h_7.65_p10k'][0])**2 +(d['h_7.65_0_0'][2]-d['h_7.65_p10k'][2])**2 )**0.5

dm1=abs(d[keys[1]][0]-d[keys[2]][0])


dm2=abs(d[keys[2]][0]-d[keys[3]][0])
dp2=abs(d[keys[3]][0]-d[keys[4]][0])



theta_dp1=np.arctan((dp1*4.63e-3)/238. )
theta_dm1=np.arctan((dm1*4.63e-3)/238. )


theta_dp2=np.arctan((dp2*4.63e-3/238.) )
theta_dm2=np.arctan((dm2*4.63e-3)/238. )

        

print("dp1=",dp1*4.6e-3, " theta dp1=",theta_dp1," theta/step= [micro rad]",(theta_dp1/10000.)*1.e6)
print("dm1=",dm1*4.6e-3, " theta dm1=",theta_dm1," theta/step= [micro rad]",(theta_dm1/10000.)*1.e6)

print('\n')
        
print("dp2=",dp2*4.6e-3, " theta dp2=",theta_dp2," theta/step= [micro rad]",(theta_dp2/10000.)*1.e6)
print("dm2=",dm2*4.6e-3, " theta dm2=",theta_dm2," theta/step= [micro rad]",(theta_dm2/10000.)*1.e6)



plt.legend()

plt.show()


