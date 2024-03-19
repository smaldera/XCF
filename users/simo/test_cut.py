import numpy as np


x=np.array([0,1,2,3,4,5])
y=np.array([0,1,2,3,4,5])
w=np.array([0,10,20,30,40,50])


print("w=",w)
print("x=",x)
print("y=",y)


mycut=w>20

print("mycut=",mycut)
print("w[mycyt]=",w[mycut])

pixcut=np.logical_not(np.logical_and(x==4,y==5))
print (pixcut)

mycut=np.logical_and(mycut,pixcut)
print("mycut=",mycut)
print("w[mycyt]=",w[mycut])
