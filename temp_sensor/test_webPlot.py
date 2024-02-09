from matplotlib import pyplot as plt
import mpld3
import numpy as np
import datetime as dt

filename='test_sensorData.txt'
f=open(filename)

time=[]
temp=[]
hum=[]

for line in f:
    print(line[:-1].split())

    time.append(dt.datetime.fromtimestamp(  (float(line[:-1].split()[0]))   ))
    temp.append(float(line[:-1].split()[1]))
    hum.append(float(line[:-1].split()[2])  )
    #print('time=',time," temp= ",temp,' humidity= ',hum)


fig = plt.figure(figsize = (10,6))
plt.plot(time,temp,'or-')
#plt.show()
html_str = mpld3.fig_to_html(fig)
Html_file= open("index.html","w")
Html_file.write(html_str)
Html_file.close()


