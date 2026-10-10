from flask import Flask, render_template
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
import numpy as np
import random
import datetime as dt
import glob
app = Flask(__name__)



def create_plots(dir_path):

    fList = glob.glob(dir_path + "/temp*.txt")

    time=[]
    temp=[]
    hum=[]
    
    for filename in fList:
        f=open(filename)
    
        for line in f:
           print(line[:-1].split())

           time.append(dt.datetime.fromtimestamp(  (float(line[:-1].split()[0]))   ))
           temp.append(float(line[:-1].split()[1]))
           hum.append(float(line[:-1].split()[2])  )
           #print('time=',time," temp= ",temp,' humidity= ',hum)

    return time,temp,hum   





@app.route('/', methods=['GET', 'POST'])
def index():

    dir_path='/home/xcf/XCF/temp_data/'
    time,temp,hum  = create_plots(dir_path)

    # Create a Bokeh scatter plot
    plot = figure(title='temperature', tools='pan,box_zoom,reset', width=900, height=300, x_axis_type="datetime")
    plot.circle(time,temp,size=10,   color="navy",   alpha=0.5 )
    plot.xaxis.axis_label="Time"
    plot.yaxis.axis_label='C'

    plotH = figure(title='humidity', tools='pan,box_zoom,reset', width=900, height=300, x_axis_type="datetime")
    plotH.circle(time,hum,size=10,   color="red",   alpha=0.5 )
    plotH.xaxis.axis_label="Time"
    plotH.yaxis.axis_label='%'

    # Embed the plot components
    script, div = components(plot)
    scriptH, divH = components(plotH)
    

    # Render the HTML template with the Bokeh plot components
    return render_template( template_name_or_list='index_bokeh2.html', script=[script,scriptH], div=[div,divH],resources=CDN.render())

if __name__ == '__main__':
   # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
