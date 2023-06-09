
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
      
    



xc_0=1434.455
yc_0=1790.625

xc_1=1280.505 # +- 0.006
yc_1=1784.147 #+- 0.007


xc_2=1141.85 #+- 0.007
yc_2= 1789.044 #+- 0.007

x_m10k=1862.12 #+- 0.006
y_m10k=1843.42 # +- 0.007

xc_3=1076.913 #+- 0.007
yc_3= 1798.704 #+- 0.007

x_p10k=309.356 #+- 0.006
y_p10k=1834.024 # +- 0.007

xc_4=1012.636 #+- 0.007
yc_4= 1806.257 #+- 0.007

#rutoto tutto!
theta=np.arctan(0.0)
print("theta=",theta)
"""
xc_0,yc_0=apply_rot(xc_0,yc_0,theta)
xc_2,yc_2=apply_rot(xc_2,yc_2,theta)
xc_3,yc_3=apply_rot(xc_3,yc_3,theta)
xc_4,yc_4=apply_rot(xc_4,yc_4,theta)
x_m10k,y_m10k=apply_rot(x_m10k,y_m10k,theta)
x_p10k,y_p10k=apply_rot(x_p10k,y_p10k,theta)
"""

d1=distance_mm(xc_2,yc_2, x_m10k,y_m10k) # distanza 1 centro, m10k
d2=distance_mm(x_m10k,y_m10k,xc_3,yc_3  )# distanza -10k , secondo centro
d3=distance_mm(x_p10k,y_p10k,xc_3,yc_3  ) # dist secondo centro p10k
d4=distance_mm(x_p10k,y_p10k,xc_4,yc_4  ) #dist p10k - centro finale


print("d1= ",d1,"  CENTRO-m10k ")
print("d2= ",d2,"  m10k - centro2  ")
print("d3= ",d3,"  centro2-p10k  ")
print("d4= ",d4,"  p10k - centro3  ")

print("d1- d2: m10k +10k ",d1-d2)
print("d3- d4: p10k-10k ",d3-d4)








fig=plt.figure(1)

plt.plot(xc_0,yc_0,'*', label='c0')
plt.plot(xc_1,yc_1,'*', label='c -20k+20k')
plt.plot(xc_2,yc_2,'*', label='c -20k + 20k')


plt.plot(xc_2,yc_2,'o', label='starting center')
plt.plot(x_m10k,y_m10k, 'o',label='-10k')

plt.plot(xc_3,yc_3,'o', label='center -10k + 10k')

plt.plot(x_p10k,y_p10k,'o', label='+10k')

plt.plot(xc_4,yc_4,'o', label='center -10k + 10k +10k -10k ')


plt.legend()

plt.show()

