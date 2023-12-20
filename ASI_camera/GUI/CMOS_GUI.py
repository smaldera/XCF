import PySimpleGUI as sg
import os.path
import os
from datetime import date
import subprocess
import time
import argparse
import numbers
import shutil
import zwoasi
from zwoasi import Camera
from cmos_pedestal2 import bg_map
from utils_v2 import read_image
from utils_v2 import plot_image
from utils_v2 import isto_all

#
# # PROSSIMI STEP
# AGGIUNGERE LA POSSIBILITà DI VEDERE I PLOT IN ANALYS E CHE NON VENGANO PUSHATI QUANDO SI FA L'ANALISI
# MODIFICARE IL PROCESSO DI ANALISI IN MODO DA RENDERE POSSIBILE L'ANALISI IN BKG COSì CHE NON SI BLOCCHI
# AGGIUNGERE BARRE DI CARICAMENTO PER QUANTO RIGUARDA LA CREAZIONE DEL PIEDISTALLO E L'ANALISI // fatto male
# INTEGRAZIONE LIBRERIE ZWO

#default values
nCore =3
xyRebin =20
sigma = 10
cluster = 10
NoClustering = True
NoEvent = True
Raw = False
Eps =1.5

file_types = [("JPEG (*.jpg)", "*.jpg", "*.png")]
id = None


def keep_files(directory, files_to_keep):
    # Get a list of all files in the directory
    all_files = os.listdir(directory)

    # Iterate through all files and delete those not in files_to_keep
    for filename in all_files:
        file_path = os.path.join(directory, filename)
        if filename not in files_to_keep:
            try:
                os.remove(file_path)
                print(f"Removed: {filename}")
            except Exception as e:
                print(f"Error removing {filename}: {e}")

def Pedestal(path_to_bkg): #Accede allo script pedestal.py
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument('-in', '--inFile', type=str, help='txt file with list of FITS files', default=path_to_bkg)
    parser.add_argument('-path', type=str, help='path to the dir for images', default=path_to_bkg)
    args = parser.parse_args()
    bg_shots_path = args.inFile
    bg_map(bg_shots_path, bg_shots_path + '/mean_ped.fits', bg_shots_path + '/std_ped.fits', args.path)

def Rm_Fits_BKG(path_to_fits): #Cancella i file .FIT nella cartella e la cartella
    #shutil.rmtree(path_to_fits)
    files_to_keep = ["std_ped.fits", "mean_ped.fits"]
    keep_files(path_to_fits, files_to_keep)

def Rm_Fits_Analy(path_to_fits):  # Cancella i file .FIT nella cartella e la cartella
    # shutil.rmtree(path_to_fits)
    files_to_keep = ["spectrum_all_raw_pixCut10.0sigma5_parallel.npz", "spectrum_all_ZeroSupp_pixCut10.0sigma5_CLUcut_10.0sigma_parallel.npz","spectrum_all_eps1.5_pixCut10.0sigma5_CLUcut_10.0sigma_parallel.npz","cluSizes_spectrum_pixCut10.0sigma5_parallel.npz","imageCUL_pixCut10.0sigma5_CLUcut_10.0sigma_parallel.fits","imageRaw_pixCut10.0sigma5_parallel.fits","events_list_pixCut10.0sigma5_CLUcut_10.0sigma.npz"]
    keep_files(path_to_fits, files_to_keep)

def Analyze(path_to_fit, path_to_bkg, cores, rebins, sigma, cluster, clu, event, raw, eps): #Accede allo script analyze_v2Parallel.py

    script_path = 'analyze_v2Parallel.py'
    script_parameters = [' -in ' + path_to_fit+'/', ' -bkg ' + path_to_bkg+'/', ' --n_jobs ' + str(cores), ' --xyrebin ' + str(rebins), ' --pix_cut_sigma ' + str(sigma), ' --clu_cut_sigma ' + str(cluster), ' --myeps ' + str(eps)]

    if clu == False:
        script_parameters.append(' --no_clustering ')
    if event == False:
        script_parameters.append(' --no_eventlist ')
    if raw == True:
        script_parameters.append(' --make_rawspectrum ')


    try:
        # Run the script with parameters using subprocess
        command = f'python3 "{script_path}" {" ".join(script_parameters)}'
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the script: {e}")



TBackground = [ #Prima Tab per il calcolo del BackGround
    [  
        sg.Text("Bkg folder  ",tooltip="Path to .FITS files"),
        sg.In(size=(45, 1), enable_events=True, key="_BKG_FOLDER_"),
        sg.FolderBrowse(),
    ],
    [

        sg.Checkbox('Delete .FIT files',key='_BKG_FITS_', tooltip=".FITS will be removed at the end")
    ],
    [
        sg.Button('Make Pedestal',key='_NOISE_',tooltip="Background Mean and RMS"),
        sg.Button('Show mean plots',key='_PLOT_MEAN_',tooltip="See them plots"),
        sg.Button('Show std plots',key='_PLOT_STD_',tooltip="See them plots"),
    ],
    #DEVE ESSERE AGGIUNTA UNA BOX PER VISUALIZZARE I PLOT DELLO SCRIPT (magari con uno slider)
    #IDEA: faccio salvare i plot e poi li leggo dalla cartella...poco pratico, ma va bene per cominciare
]

TAnalyze = [ #Seconda tab per l'analisi del segnale
    [  
        sg.Text("Bkg folder    ",tooltip="Path to Bkg files"),
        sg.In(size=(45, 1), enable_events=True, key="_BKG_FOLDER_A_"),
        sg.FolderBrowse(),
    ],
    [  
        sg.Text("Data folder   ",tooltip="Path to .FIT files"),
        sg.In(size=(45, 1), enable_events=True, key="_FIT_FOLDER_"),
        sg.FolderBrowse(),
    ],
    [#InputBox per i parametri dello script
        sg.Text('N° Process at simultaneous time'), #non si vede bene il testo!
        sg.In( 3,    key='_CORE_', enable_events=True,   tooltip="PC cores used",                size=(10, 1)),
        sg.Text('XY Rebin'),        
        sg.In(20,    key='_REBIN_', enable_events=True,  tooltip="Rebin XY",                     size=(10, 1)),
        sg.Text('Sigma Cut'),       
        sg.In(10,    key='_SIGMA_', enable_events=True,  tooltip="Cuts based on n*RMS",          size=(10, 1)),
        sg.Text('Cluster Cut'),     
        sg.In(10,    key='_CLUSTER_', enable_events=True,tooltip="Cuts based on mean + n*RMS",   size=(10, 1)),
        sg.Text('EPS parameter'),       
        sg.In(1.5,    key='_EPS_', enable_events=True,    tooltip="Allows DBSCAN eps parameter",  size=(10, 1)),
    ],
    [#Checkbox per le opzioni aggiuntive
        sg.Checkbox('Clustering',  key='_CLUSTERING_', tooltip="Clustering On/Off"      , default=True),
        sg.Checkbox('Event List',  key='_EVENTS_',     tooltip="Makes the Event List"   , default=True),
        sg.Checkbox('Raw Spectrum',key='_RAW_',        tooltip="Plots the Raw Spectrum" , default=False),
    ],
    [#Start dello script
        sg.Button('Start Analysis',     key='_AN_START_',    tooltip="Data anlyser"),
        sg.Checkbox('Delete .FIT files',key='_SIGNAL_FIT_', tooltip=".FIT will be removed at the end")
    ],
    #DEVE ESSERE AGGIUNTA UNA BOX PER VISUALIZZARE I PLOT DELLO SCRIPT (magari con uno slider)
]


TCamera = [ #Terza tab per visualizzare la lista eventi
    [
        sg.Text("Camera info:"),

    ],
    [

    ],
]






# ----- Full layout -----
Tab2 = sg.Tab("Background", TBackground)
Tab3 = sg.Tab("Analyze", TAnalyze)
Tab1 = sg.Tab("Camera", TCamera)

TabGrp = sg.TabGroup([[Tab1, Tab2,Tab3]], tab_location='centertop',
                     selected_title_color='Green', selected_background_color='Gray', border_width=3)
window = sg.Window("CMOS analyzer V0.1.1", [[TabGrp]])

# ----- Commands -----
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    #-------------------------------------BACKGROUND-----------------------------------    
    if event == "_BKG_FOLDER_":
        if os.path.exists(values["_BKG_FOLDER_"]):
            bkg_folder = values["_BKG_FOLDER_"]
        # else:
        #     sg.popup_annoying('Dir not found')


    if event == "_PLOT_MEAN_":
        try:
            isto_all(read_image(bkg_folder + '/mean_ped.fits'))
            plot_image(read_image(bkg_folder + '/mean_ped.fits'))

        except NameError:
            sg.popup('You sure them plots are available?')
    if event == "_PLOT_STD_":
        try:
            isto_all(read_image(bkg_folder + '/std_ped.fits'))
            plot_image(read_image(bkg_folder + '/std_ped.fits'))
        except NameError:
            sg.popup('You sure them plots are available?')
    if event == "_NOISE_":
        if os.path.exists(values["_BKG_FOLDER_"]):
            try:
                Pedestal(bkg_folder)
                sg.popup('BKG ready to use')
                if values['_BKG_FITS_'] == True:
                    try:
                        Rm_Fits_BKG(bkg_folder)
                        sg.popup('BKG have been removed')
                    except NameError:
                        sg.popup('cannot remove files')
            except NameError:
                sg.popup('An ERROR OCCURRED: cannot lauch pedestal()')

        else:
            sg.popup_annoying('Dir not found')


    #-------------------------------------ANALYSE-------------------------------------

    if event == "_BKG_FOLDER_A_":
        if os.path.exists(values["_BKG_FOLDER_A_"]):
            bkg_folder_a = values["_BKG_FOLDER_A_"]

    if event == "_FIT_FOLDER_": #Folder with .FIT files
        if os.path.exists(values["_FIT_FOLDER_"]):
            fit_folder = values["_FIT_FOLDER_"]
        # else:
        #     sg.popup('Dir not found')

    if event == "_CORE_":
        if (values['_CORE_'] in ('0123456789')):   #Rebin number
            nCore = values["_CORE_"]
        else:
            sg.popup("Only digit allowed")


    if event == "_REBIN_":
        if (values['_REBIN_']in ('0123456789')):   #Rebin number
            xyRebin = values["_REBIN_"]
        else:
            sg.popup("Only idigit allowed")


    if event == "_SIGMA_":
        if (values['_SIGMA_'] in ('0123456789')):   #Number of sigma used for cuts
            sigma = values["_SIGMA_"]
        else:
            sg.popup("Only idigit allowed")

    if event == "_CLUSTER_":
        if (values['_CLUSTER_']in ('0123456789')): #Minimum number of pixel per cluster
            cluster = int(values["_CLUSTER_"])
        else:
            sg.popup("Only idigit allowed")

    if event == "_EPS_":
        if (values['_EPS_'] in ('0123456789.')):    #EPS parameter
            Eps = values["_EPS_"]
        else:
            sg.popup("Only idigit allowed (i.e. 1.56)")

    if values['_CLUSTERING_']==False:
        NoClustering = False
    if values['_EVENTS_']==False:
        NoEvent = False
    if values['_RAW_']==True:
        Raw = True
    if event == "_AN_START_":
        try:
            fit_folder
            try:
                bkg_folder_a
                try:
                    Analyze(fit_folder, bkg_folder_a, nCore, xyRebin, sigma, cluster, NoClustering, NoEvent, Raw, Eps)
                    if values['_SIGNAL_FIT_'] == True:
                        Rm_Fits_Analy(fit_folder)
                        sg.popup('fits have been removed')
                except NameError:
                    sg.popup("cannot lunch analyza {e}")
            except NameError:
                sg.popup('location of bkg is not defined.')
        except NameError:
            sg.popup('location of data is not defined.')
    #if event == "_AN_START_":
    #     Analyze(fit_folder, bkg_folder_a, nCore, xyRebin, sigma, cluster, NoClustering, NoEvent, Raw, Eps)




window.close()