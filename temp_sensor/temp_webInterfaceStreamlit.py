# USAGE:  stramlit run app_v3.py
# credits: Gabriele GT

import streamlit as st
import numpy as np
import plotly.express as px
import os
import glob
from datetime import datetime
import plotly.graph_objects as go



st.title("XCF - Monitor temperatura (Arduino)")

#  directory
#directory = r"C:/Users/gabri/OneDrive/Fisica/STAGE/dati_Arduino_new"
directory = "/home/xcf/XCF/temp_data/"


time = []
timeTk = []

temp_SHT4 = []
hum = []
temp_termo = []

files_list = glob.glob(os.path.join(directory, "temp_sensorData_*.txt"))

print(files_list)

for myfile in files_list:
#if files_list:
    file_path = myfile #files_list[0]
    print("leggo ",myfile)
    with open(file_path, 'r') as f:
        for line in f:
            colonne = line.split()
            
            # Prendo i dati solo se la riga ha almeno 4 colonne (sensore + termocoppia)
            if len(colonne) < 4:
                #continue
                time.append(datetime.fromtimestamp(float(colonne[0])))
                temp_SHT4.append(float(colonne[1]))
                hum.append(float(colonne[2]))

            
            else:
                #time.append(datetime.fromtimestamp(float(colonne[0])))
                #temp_SHT4.append(float(colonne[1]))
                #hum.append(float(colonne[2]))
                temp_termo.append(float(colonne[3]))
                timeTk.append(datetime.fromtimestamp(float(colonne[0])))
                time.append(datetime.fromtimestamp(float(colonne[0])))
                temp_SHT4.append(float(colonne[1]))
                hum.append(float(colonne[2]))

# Conversione in array
if timeTk:
    time_arr = np.array(time)
    timeTk_arr=np.array(timeTk)
    temp_SHT4_arr = np.array(temp_SHT4)
    hum_arr = np.array(hum)
    temp_termo_arr = np.array(temp_termo)

    #non so se è comodo il trend ultimi 10 elementi
    st.subheader("Medie ultime 10 misure")
    last_10_temp = temp_SHT4_arr[-10:]
    last_10_hum = hum_arr[-10:]
    last_10_termo = temp_termo_arr[-10:]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Temperatura SHT4", f"{np.mean(last_10_temp):.2f}°C")
    with c2:
        st.metric("Temperatura Termocoppia", f"{np.mean(last_10_termo):.2f}°C")
    with c3:
        st.metric("Media Umidità", f"{np.mean(last_10_hum):.1f}%")





    #con dragmode seleziono il modo in cui spostare il grafico (può essere 'pan', oppure 'zoom')
    # pan--> traslazione rigida; zoom-->zoomma
    # use_container_width=True --> usa tutta la larghezza disponibile



    # Plot Temperatura SHT4
    fig_t = px.scatter(x=time_arr, y=temp_SHT4_arr, title='Temperatura SHT4 (°C)')
    #fig_t.update_layout(xaxis_rangeslider_visible=True, dragmode='zoom')
    st.plotly_chart(fig_t, use_container_width=True)

    # Plot Termocoppia
    fig_tc = px.scatter(x=timeTk_arr, y=temp_termo_arr, title='Temperatura Termocoppia (°C)')
    #fig_tc.update_layout(xaxis_rangeslider_visible=True, dragmode='zoom')
    st.plotly_chart(fig_tc, use_container_width=True)

    # Plot Umidità
    fig_h = px.line(x=time_arr, y=hum_arr, title='Umidità (%)')
   # fig_h.update_layout(xaxis_rangeslider_visible=True, dragmode='zoom')
    st.plotly_chart(fig_h, use_container_width=True)







# per plot unico bisogna usare plotly.graph_objects (--> go)
fig = go.Figure()

fig.add_trace(go.Scatter(x=time_arr,y=temp_SHT4_arr,mode='lines+markers',name="SHT4"))
fig.add_trace(go.Scatter(x=time_arr,y=temp_termo_arr,mode='lines+markers',name="Termocoppia"))
#fig.update_layout(title="Temperature SHT4 e Termocoppia",xaxis_rangeslider_visible=True,dragmode='pan')
st.plotly_chart(fig, use_container_width=True)
