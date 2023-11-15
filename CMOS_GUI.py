import PySimpleGUI as sg
import os.path
import os
from datetime import date


file_types = [("JPEG (*.jpg)", "*.jpg", "*.png")]

def Pedestal(path_to_bkg, name): #Accede allo script pedestal.py
    os.system('python3 /home/frassi/Desktop/XCF-main/ASI_camera/analysis_simo/pedestal.py -in '+path_to_bkg+
              ' -path '+path_to_bkg+' -name '+name)
def Rm_Fits(path_to_fits): #Cancella i file .FIT nella cartella
    os.remove(path_to_fits+'/*.FIT')
def Analyze(path_to_fit, path_to_bkg, cores, rebins, sigma, cluster, clu, event, raw, eps): #Accede allo script analyze_v2Parallel.py
    os.system('python3 /home/frassi/Desktop/XCF-main/ASI_camera/analysis_simo/analyze_v2Parallel.py -in '+path_to_fit+
              " -bkg "+path_to_bkg+' --n_jobs '+cores+' --xyrebin '+rebins+' --pix_cut_sigma '+sigma+
              ' --clu_cut_sigma '+cluster+' --no_clustering '+clu+' --no_eventlist '+event+' --make_rawspectrum '
              +raw+' --myeps '+eps)


TBackground = [ #Prima Tab per il calcolo del BackGround
    [  
        sg.Text("Bkg folder  ",tooltip="Path to .FITS files"),
        sg.In(size=(45, 1), enable_events=True, key="_BKG_FOLDER_"),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("File's name",tooltip="BKG's plots name"),
        sg.In(date.today().strftime("%d_%m_%Y_%H_%M"), size=(45, 1), enable_events=True, key="_BKG_NAME_"),
        sg.Checkbox('Delete .FIT files',key='_BKG_FITS_', tooltip=".FITS will be remooved at the end")
    ],
    [
        sg.Button('Make Pedestal',key='_NOISE_',tooltip="Background Mean and RMS"),
    ],
    [
        sg.Image(key="_IMAGE_")
    ],
    #DEVE ESSERE AGGIUNTA UNA BOX PER VISUALIZZARE I PLOT DELLO SCRIPT (magari con uno slider)
    #IDEA: faccio salvare i plot e poi li leggo dalla cartella...poco pratico, ma va bene per cominciare
]

TAnalyze = [ #Seconda tab per l'analisi del segnale
    [  
        sg.Text("Bkg folder    ",tooltip="Path to Bkg files"),
        sg.In(size=(45, 1), enable_events=True, key="_BKG_FOLDER_"),
        sg.FolderBrowse(),
    ],
    [  
        sg.Text("Data folder   ",tooltip="Path to .FIT files"),
        sg.In(size=(45, 1), enable_events=True, key="_FIT_FOLDER_"),
        sg.FolderBrowse(),
    ],
    [#InputBox per i parametri dello script
        sg.Text('N° Cores (',os.cpu_count(),'max)'), #non si vede bene il testo!
        sg.In('2',    key='_CORE_',   tooltip="PC cores used",                size=(10, 1)),
        sg.Text('XY Rebin'),        
        sg.In('20',    key='_REBIN_',  tooltip="Rebin XY",                     size=(10, 1)),
        sg.Text('Sigma Cut'),       
        sg.In('10',    key='_SIGMA_',  tooltip="Cuts based on n*RMS",          size=(10, 1)),
        sg.Text('Cluster Cut'),     
        sg.In('10',    key='_CLUSTER_',tooltip="Cuts based on mean + n*RMS",   size=(10, 1)),
        sg.Text('EPS parameter'),       
        sg.In('1.5',    key='_EPS_',    tooltip="Allows DBSCAN eps parameter",  size=(10, 1)),
    ],
    [#Checkbox per le opzioni aggiuntive
        sg.Checkbox('Clustering',  key='_CLUSTERING_', tooltip="Clustering On/Off"      , default=True),
        sg.Checkbox('Event List',  key='_EVENTS_',     tooltip="Makes the Event List"   , default=True),
        sg.Checkbox('Raw Spectrum',key='_RAW_',        tooltip="Plots the Raw Spectrum" , default=False),
    ],
    [#Start dello script
        sg.Button('Start Analysis',     key='_AN_START_',    tooltip="Data anlyser"),
        sg.Checkbox('Delete .FIT files',key='_SIGNAL_FIT_', tooltip=".FIT will be remooved at the end")
    ],
    #DEVE ESSERE AGGIUNTA UNA BOX PER VISUALIZZARE I PLOT DELLO SCRIPT (magari con uno slider)
]
#DEVO INSERIRE LA TERZA TAB PER READ_EVENTLIST.PY!!!!


# ----- Full layout -----
Tab1 = sg.Tab("Background", TBackground)
Tab2 = sg.Tab("Analyze", TAnalyze)
TabGrp = sg.TabGroup([[Tab1, Tab2]], tab_location='centertop', 
                     selected_title_color='Green', selected_background_color='Gray', border_width=3)
window = sg.Window("CMOS analyzer V0.1", [[TabGrp]])


# ----- Commands -----
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    #-------------------------------------BACKGROUND-----------------------------------    
    if event == "_BKG_FOLDER_":
        if os.path.exists(values["_BKG_FOLDER_"]):
            bkg_folder = values["_BKG_FOLDER_"]
        else:
            sg.popup_annoying('Dir not found')
    if event == "_BKG_NAME_":
        bkg_name = values['_BKG_NAME_']
    if event == "_NOISE_" and os.path.exists(values["_BKG_FOLDER_"]):
        Pedestal(bkg_folder, bkg_name)
        if values['_BKG_FITS_']==True:
            Rm_Fits(bkg_folder)

    #-------------------------------------ANALYSE-------------------------------------
    if event == "_FIT_FOLDER_": #Folder with .FIT files
        fit_folder = values["_FIT_FOLDER_"]

    if event == "_CORE_":
        if (values['_CORE_'][-1] in ('0123456789')):  #Number of cores enabled (need to be integer)
            if values['CORE'][-1]<=os.count() and values['CORE'][-1]>=1:
                nCore = values["_CORE_"]
        else:
            sg.popup("Only digits allowed")
            window['_CORE_'].update(values['_CORE_'][:-1])

    if event == "_REBIN_":
        if (values['_REBIN_'][-1] in ('0123456789')):   #Rebin number
            xyRebin = int(values["_REBIN_"])
        else:
            sg.popup("Only idigit allowed")
            window['_REBIN_'].update(values['_REBIN_'][:-1])

    if event == "_SIGMA_":
        if (values['_SIGMA_'][-1] in ('0123456789')):   #Number of sigma used for cuts
            sigma = int(values["_SIGMA_"])
        else:
            sg.popup("Only idigit allowed")
            window['_SIGMA_'].update(values['_SIGMA_'][:-1])

    if event == "_CLUSTER_":
        if (values['_CLUSTER_'][-1] in ('0123456789')): #Minimum number of pixel per cluster
            cluster = int(values["_CLUSTER_"])
        else:
            sg.popup("Only idigit allowed")
            window['_CLUSTER_'].update(values['_CLUSTER_'][:-1])

    if event == "_EPS_":
        if (values['_EPS_'][-1] in ('0123456789.')):    #EPS parameter
            Eps = values["_EPS_"]
        else:
            sg.popup("Only idigit allowed (i.e. 1.56)")
            window['_EPS_'].update(values['_EPS_'][:-1])
    
    if values['_CLUSTERING_']==False:
        NoClustering = False
    if values['_EVENTS_']==False:
        NoEvent = False
    if values['_RAW_']==True:
        Raw = True
    if event == "_AN_START_":
        Analyze(fit_folder, bkg_folder, nCore, xyRebin, sigma, cluster, NoClustering, NoEvent, Raw, Eps)
        if values['_SIGNAL_FIT_']==True:
            Rm_Fits(fit_folder)
    


window.close()