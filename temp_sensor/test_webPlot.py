from matplotlib import pyplot as plt
import mpld3
import numpy as np
import datetime as dt
#from matplotlib.widgets import Slider


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
import matplotlib.dates as mdates

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%m'))
#plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate()

# time axis sliders... nor working for now
#time_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
#time_slider = Slider(time_slider_ax, 'time', time[0], time[:0])
#def sliders_on_changed(val):
#    plt.set_xlim([val-3600*24, val+3600*24])
#    fig.canvas.draw_idle()  
#time_slider.on_changed(sliders_on_changed)


plt.show()

html_str = mpld3.fig_to_html(fig)
Html_file= open("index.html","w")
Html_file.write(html_str)
Html_file.close()


