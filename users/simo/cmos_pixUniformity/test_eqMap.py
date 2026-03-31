import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt






x=np.array([1,2,3,4,5,6,7,8,9,1,1,1,1,8,8,8])
y=np.array([1,2,3,4,5,6,7,8,9,1,1,1,1,2,2,2])
w=np.array([1,2,3,4,5,1,1,1,1,1,1,1,1,3,3,3])



print("x=",x)
print("y=",y)

counts2d,  xedges, yedges= np.histogram2d(x,y,bins=[10, 10 ],range=[[0,10],[0,10]])

counts2dW,  xedges, yedges= np.histogram2d(x,y,weights=w,bins=[10, 10 ],range=[[0,10],[0,10]])
counts2dW=counts2dW/counts2d
print("counts2d=",counts2d)

ax1=plt.subplot(111)
im=ax1.imshow((counts2d), interpolation='nearest', origin='lower',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

# cerco indici:
x_idx=np.searchsorted(xedges,x)
print("x_idx=",x_idx)

y_idx=np.searchsorted(xedges,y)
print("y_idx=",y_idx)

#corrw= counts2dW[1,1][1,1]
values = counts2d[x_idx,y_idx]

print("values=",values)
plt.show()
