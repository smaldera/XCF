from astropy.io import fits as pf
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import os

P = []
i = 0
x = []
y = []

weights = open("C:\\Users\\Acer\\Downloads\\Uni\\Tesi\\Dati\\weights.txt", "r")
pixels = open("C:\\Users\\Acer\\Downloads\\Uni\\Tesi\\Dati\\Pixels.txt", "r")

w = (weights.read())
op = w.strip('][').split(', ')
type(op)
weight = np.array(op, dtype=float)

lines = pixels.readlines()


for line in lines:
    sline = line.strip('\n')
    spline = sline.partition(',')
    print(spline)
    P.insert(i, spline)
    i = i+1
    
p = np.array(P)

for row in p:
    x.append(int(row[0]))
    y.append(int(row[2]))

xpixels = np.array(x)
ypixels = np.array(y)


%matplotlib notebook
fig, ax = plt.subplots()
ax.hist2d(xpixels, ypixels, bins = [4144, 2822], range = [[0,4144], [0,2822]], weights = weight, cmap=plt.cm.jet)

plt.figure()
plt.show()
    
     

