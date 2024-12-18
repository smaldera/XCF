import FreeSimpleGUI as sg
import os.path
import os
import subprocess
import argparse
import zwoasi as asi
import configparser
from cmos_pedestal import bg_map
from cmos_pedestal import bg_map_rt
from cmos_pedestal import capture
from utils_v2 import read_image
from utils_v2 import plot_image
from utils_v2 import isto_all
from gui_analyzer_parallel import aotr2
import multiprocessing
import glob

#-------------------------------------------------------------------------------------------------
#---------------------------------------------CONFIG FILE-----------------------------------------
def read_config(filename='config.ini'): #opens filename (config)
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def update_config(): #updates filename (config)
    config['Settings']['nCore']= str(nCore)
    config['Settings']['xyRebin'] = str(xyRebin)
    config['Settings']['sigma'] = str(sigma)
    config['Settings']['cluster'] = str(cluster)
    config['Settings']['NoClustering'] = str(NoClustering)
    config['Settings']['NoEvent'] = str(NoEvent)
    config['Settings']['Raw'] = str(Raw2)
    config['Settings']['Eps'] = str(Eps)
    config['Settings']['StoreDataIn'] = str(StoreDataIn)
    config['Settings']['SampleSize'] = str(SampleSize)
    config['Settings']['WBR'] = str(WBR)
    config['Settings']['WBB'] = str(WBB)
    config['Settings']['exposure'] = str(exposure)
    config['Settings']['gain'] = str(gain)
    config['Settings']['file_name'] = str(file_name)
    config['Settings']['xyRebin2'] = str(xyRebin2)
    config['Settings']['sigma2'] = str(sigma2)
    config['Settings']['cluster2'] = str(cluster2)
    config['Settings']['NoClustering2'] = str(NoClustering2)
    config['Settings']['NoEvent2'] = str(NoEvent2)
    config['Settings']['Raw2'] = str(Raw2)
    config['Settings']['Eps2'] = str(Eps2)
    config['Settings']['num'] = str(num)
    config['Settings']['length'] = str(length)
    config['Settings']['bkg_folder_b'] = str(bkg_folder_b)
    config['Settings']['bkg_folder'] = str(bkg_folder)
    config['Settings']['bkg_folder_a'] = str(bkg_folder_a)
    config['Settings']['fit_folder'] = str(fit_folder)
    config['Settings']['bunch'] = str(bunch)
    write_config(config)
    
def update_info(): #Updates Camera info
    num_cameras = asi.get_num_cameras()
    if num_cameras == 0:
        window['_OUTPUT_'].update("There is no camera connected")
    if num_cameras == 1:
        camera_info = asi.list_cameras()
        window['_OUTPUT_'].update(camera_info)
    if num_cameras > 1: #for the moment it ignores all other cameras
        print('Please connect only one camera at a time')
        camera_info = asi.list_cameras()
        window['_OUTPUT_'].update(camera_info)

def write_config(config, filename='config.ini'): #writes in filename (config)
    with open(filename, 'w') as configfile:
        config.write(configfile)







#-------------------------------------------------------------------------------------------------
#---------------------------------------------ANALYSIS--------------------------------------------
def Analyze(path_to_fit, path_to_bkg, cores, rebins, sigma, cluster, clu, event, raw, eps): #Launches analyze_v2Parallel.py
        script_path = 'analyze_v2Parallel.py'
        script_parameters = [' -in ' + path_to_fit+'/', 
                             ' -bkg ' + path_to_bkg+'/', 
                             ' --n_jobs ' + str(cores), 
                             ' --xyrebin ' + str(rebins), 
                             ' --pix_cut_sigma ' + str(sigma), 
                             ' --clu_cut_sigma ' + str(cluster), 
                             ' --myeps ' + str(eps)]
        print('event= ',event) 
        if clu == False:
            script_parameters.append(' --no_clustering ')
        if event == False:
            script_parameters.append(' --no_eventlist ')
        if raw == True:
            script_parameters.append(' --make_rawspectrum ')
        try:
            # Run the script with parameters using subprocess
            command = f'python3 "{script_path}" {" ".join(script_parameters)}'   #!!!!!!!!!!!! python3????
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running the script: {e}")
            
def CaptureAndAnalyze2(path, sample_size, WB_R, WB_B, EXPO, GAIN, bkg_folder_a, xyRebin, sigma, cluster, NoClustering, NoEvent, Raw, Eps, num, leng, bunch):
        print('Make event list in CaptureAndAnalyze2=',NoEvent)
        OBJ = aotr2(path, sample_size, WB_R, WB_B, EXPO, GAIN, bkg_folder_a, xyRebin, sigma, cluster, NoClustering, NoEvent,
                   Raw, Eps,num ,leng,bunch)
        try:
            OBJ.CaptureAnalyze() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            sg.popup("Analysis has been completed and files are now stored in: " + path)
        except Exception as e:
            sg.popup(f"Error in CaptureAndAnalyze2: {e}")
            print("\n There are troubles in CaptureAndAnalyze2: ", e)
            
def DataChunk(path, name, sample_size, WB_R, WB_B, EXPO, GAIN): #Tries to take a snap
    try:
        capture(name, path, sample_size, WB_R, WB_B, EXPO, GAIN)
        sg.popup("Snaps taken and saved in " + path)
    except Exception as e:
        sg.popup(f"Capture error: {e}")
        
def Pedestal(path_to_bkg): #Opens pedestal.py
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument('-in', '--inFile', type=str, help='txt file with list of FITS files', default=path_to_bkg)
    parser.add_argument('-path', type=str, help='path to the dir for images', default=path_to_bkg)
    args = parser.parse_args()
    bg_shots_path = args.inFile
    bg_map(bg_shots_path, bg_shots_path + '/mean_ped.fits', bg_shots_path + '/std_ped.fits', args.path)

def Pedestal2(path_to_bkg,sample_size,GAIN,WB_B,WB_R,EXPO): #Uses bg_map_rt
    try:
        bg_map_rt(path_to_bkg, path_to_bkg + '/mean_ped.fits', path_to_bkg + '/std_ped.fits', sample_size, GAIN, WB_B, WB_R, EXPO)
    except Exception as e:
        sg.popup(f"Error in Pedestal2: {e}")
        print("Error in Pedestal2: " , e)
        
def Rm_Fits_Files(path_to_fits): #Deletes all .FITS files from the folder
    file_list = glob.glob(os.path.join(path_to_fits, '*.FIT'))
    for file_path in file_list:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Rm_Fits_Files error: {e} in {file_path}")







if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # or 'forkserver'
    sg.theme('LightGreen')  # Choose a theme
    
    env_filename = os.getenv('ZWO_ASI_LIB')
    parser = argparse.ArgumentParser(description='Process and save images from a camera')
    parser.add_argument('filename',
                        nargs='?',
                        help='SDK library filename')
    args = parser.parse_args()

    #-----------------Initialize zwoasi with the name of the SDK library-----------------
    if args.filename:
        asi.init(args.filename)
    elif env_filename:
        asi.init(env_filename)
    else:
        print('The filename of the SDK library is required (or set ZWO_ASI_LIB environment variable with the filename)')
    file_types = [("JPEG (*.jpg)", "*.jpg", "*.png")]
    id = None
    
    #READING AND SETTING ALL THE PRE-CONFIG
    config = read_config()
    nCore = int(config['Settings']['nCore'])
    xyRebin = int(config['Settings']['xyRebin'])
    sigma = int(config['Settings']['sigma'])
    cluster = int(config['Settings']['cluster'])
    NoClustering = config['Settings'].getboolean('NoClustering')
    NoEvent = config['Settings'].getboolean('NoEvent')
    Raw = config['Settings'].getboolean('Raw')
    Eps = float(config['Settings']['Eps'])
    StoreDataIn = config['Settings']['StoreDataIn']
    SampleSize = int(config['Settings']['SampleSize'])
    WBR = int(config['Settings']['WBR'])
    WBB = int(config['Settings']['WBB'])
    exposure = int(config['Settings']['exposure'])
    gain = int(config['Settings']['gain'])
    file_name = config['Settings']['file_name']
    xyRebin2 = int(config['Settings']['xyRebin2'])
    sigma2 = int(config['Settings']['sigma2'])
    cluster2 = int(config['Settings']['cluster2'])
    NoClustering2 = config['Settings'].getboolean('NoClustering2')
    NoEvent2 = config['Settings'].getboolean('NoEvent2')
    Raw2 = config['Settings'].getboolean('Raw2')
    Eps2 = float(config['Settings']['Eps2'])
    num = int(config['Settings']['num'])
    length = int(config['Settings']['length'])
    bkg_folder_b = config['Settings']['bkg_folder_b']
    bkg_folder = config['Settings']['bkg_folder']
    bkg_folder_a = config['Settings']['bkg_folder_a']
    fit_folder = config['Settings']['fit_folder']
    bunch = int(config['Settings']['bunch'])



    #-------------------------------------------------------------------------------------------------
    #---------------------------------------------GUI-------------------------------------------------
    TAnalyze = [ #Tab for Signal
        [
            sg.Text('Background Map', font=('Helvetica', 20), text_color='Green'),
        ],
        [
            sg.Text("Bkg folder  ",tooltip="Path to .FITS files"),
            sg.In(bkg_folder,size=(35, 1), enable_events=True, key="_BKG_FOLDER_"),
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
        #WE MAY NEED A BOX TO DISPLAY ONLINE GRAPH 
        [
            sg.Text('Analyze collected data', font=('Helvetica', 20), text_color='Green'),
        ],
        [
            sg.Text("Bkg folder    ",tooltip="Path to Bkg files"),
            sg.In(bkg_folder_a,size=(35, 1), enable_events=True, key="_BKG_FOLDER_A_"),
            sg.FolderBrowse(),
        ],
        [
            sg.Text("Data folder   ",tooltip="Path to .FIT files"),
            sg.In(fit_folder,size=(35, 1), enable_events=True, key="_FIT_FOLDER_"),
            sg.FolderBrowse(),
        ],
        [#InputBox for parameters
            sg.Text('Simoultaneous process '),
            sg.In( nCore, key='_CORE_', enable_events=True, tooltip="PC cores used", size=(5, 1)),
            sg.Text('XY Rebin'),
            sg.In(xyRebin, key='_REBIN_', enable_events=True, tooltip="Rebin XY", size=(5, 1)),
        ],
        [
            sg.Text('Sigma Cut'),
            sg.In(sigma, key='_SIGMA_', enable_events=True, tooltip="Cuts based on n*RMS", size=(5, 1)),
            sg.Text('Cluster Cut'),
            sg.In(cluster, key='_CLUSTER_', enable_events=True,tooltip="Cuts based on mean + n*RMS", size=(5, 1)),
            sg.Text('EPS parameter'),
            sg.In(Eps, key='_EPS_', enable_events=True, tooltip="Allows DBSCAN eps parameter", size=(5, 1)),
        ],
        [#Extra-options Checkbox 
            sg.Checkbox('Clustering', key='_CLUSTERING_',enable_events=True, tooltip="Clustering On/Off", default=NoClustering),
            sg.Checkbox('Event List', key='_EVENTS_', enable_events=True, tooltip="Makes the Event List", default=NoEvent),
            sg.Checkbox('Raw Spectrum', key='_RAW_', enable_events=True, tooltip="Plots the Raw Spectrum", default=Raw),
        ],
        [#Start acquiring
            sg.Button('Start Analysis', key='_AN_START_', tooltip="Data anlyser"),
            sg.Checkbox('Delete .FIT files', key='_SIGNAL_FIT_', tooltip=".FIT will be removed at the end")
        ],
    ]

    TCamera = [ #Prima tab per cattura e analizza
        [       ],
        [
            sg.Text("Camera info:"),
            sg.Text(" Press update to check",size=(20,1), key='_OUTPUT_'),
            sg.Button('Update', key='_CAMERA_UPDATE_', tooltip="Check if a ZWO ASI camera is connected")
        ],
        [       ],
        [
            sg.Text('Camera settings', font=('Helvetica', 15), text_color='Green')
        ],
        [       ],
        [   
            sg.Text("Destination folder   "),
            sg.In(StoreDataIn,size=(15, 1), enable_events=True, key="_DATA_FOLDER_"),
            sg.FolderBrowse(),
        ],
        [   
            sg.Text("Exposure (mu-s)  ", tooltip="how long ?"),
            sg.In(exposure,size=(5, 1), enable_events=True, key="_EXPOSURE_"),
            sg.Text("   Exposure  gain        ", tooltip=""),
            sg.In(gain,size=(5, 1), enable_events=True, key="_GAIN_"),
        ],
        [   
            sg.Text("White Balance Red", tooltip="?"),
            sg.In(WBR,size=(5, 1), enable_events=True, key="_WB_R_"),
            sg.Text("   White Balance Blue ", tooltip=""),
            sg.In(WBB,size=(5, 1), enable_events=True, key="_WB_B_"),
        ],
        [   
            sg.Text("Number of samples", tooltip="HOW MANY OF THEM FITS?"),
            sg.In(SampleSize,size=(5, 1), enable_events=True, key="_SAMPLE_SIZE_"),
            sg.Text("   Files name              ", tooltip="?"),
            sg.In(file_name,size=(5, 1), enable_events=True, key="_FILE_NAME_"),
        ],
        [
            sg.Button('Capture Data', key='_COLLECT_DATA_', tooltip="This will ONLY collect data"),
            sg.Button('Capture and make BKG', key='_NOISE2_', tooltip="This will collect data and make BKG", )
        ],
        [
            sg.Text(" "),
        ],
        [
            sg.Text('Analysis Settings', font=('Helvetica', 15), text_color='Green'),
        ],
        [       ],
        [
            sg.Text("Note: previous section needs to be filled to set camera values",
                    tooltip="capture and analyze data.     "),
        ],
        [
            sg.Text("BKG folder ", tooltip="Folder containing std_ped.fits and mean_ped.fits"),
            sg.In(bkg_folder_b,size=(15, 1), enable_events=True, key="_BKG_FOLDER_2_"),
            sg.FolderBrowse(),
        ],
        [
            sg.Text('N. analyzer'),
            sg.In(num, key='_NUM_', enable_events=True, tooltip="number of parallel process to analyze data", 
                  size=(5, 1)),
            sg.Text('Max queue        '),
            sg.In(length, key='_LEN_', enable_events=True, tooltip="max lenght for the analyzer queue. the queue is stored in the RAM. 100 items is approximately 2.3GB ", 
                  size=(5, 1)),
        ],
        [
            sg.Text('XY Rebin    '),
            sg.In(xyRebin2, key='_REBIN2_', enable_events=True,  tooltip="Rebin XY", size=(5, 1)),
            sg.Text('Sigma Cut        '),
            sg.In(sigma2, key='_SIGMA2_', enable_events=True,  tooltip="Cuts based on n*RMS", size=(5, 1)),
        ],
        [
            sg.Text('Cluster Cut '),
            sg.In(cluster2, key='_CLUSTER2_', enable_events=True,tooltip="Cuts based on mean + n*RMS", size=(5, 1)),
            sg.Text('EPS parameter'),
            sg.In(Eps2, key='_EPS2_', enable_events=True, tooltip="Allows DBSCAN eps parameter", size=(5, 1)),
        ],
        [
            sg.Checkbox('Clustering',  key='_CLUSTERING2_',enable_events=True, tooltip="Clustering On/Off"      , default=NoClustering2),
            sg.Checkbox('Event List',  key='_EVENTS2_',  enable_events=True,   tooltip="Makes the Event List"   , default=NoEvent2),
            sg.Checkbox('Raw Spectrum',key='_RAW2_',   enable_events=True,     tooltip="Plots the Raw Spectrum" , default=Raw2),
        ],
        [
            sg.Text('Save progress: '),
            sg.In(bunch,    key='_BUNCH_', enable_events=True,    tooltip="the script saves automatically every five minutes"
                                                                          "\n but can saves also after a certain amount of snaps \n"
                                                                          "which ever comes first",  size=(5, 1)),
        ],
        [
            #sg.Button('Collect and Analyze Real Time', key='_CAPTURE_AND_ANALYZE_', tooltip="Start data acquiring and analysis"),
            sg.Button('Collect and Analyze', key='_CAPTURE_AND_ANALYZE_PARALLEL_',
                      tooltip="Start data acquiring and analysis")
        ],
    ]




    #FULL GUI LAYOUT
    Tab3 = sg.Tab("bkg and analysis", TAnalyze)
    Tab1 = sg.Tab(" Camera control", TCamera)
    TabGrp = sg.TabGroup([[Tab1,Tab3]], tab_location='centertop',
                         selected_title_color='Green', selected_background_color='Gray', border_width=3)
    window = sg.Window("CMOS controller and data analizer", [[TabGrp]])




    #-------------------------------------------------------------------------------------------------
    #---------------------------------------------COMMANDS--------------------------------------------
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        #--------------------------------------------------------------------------------------
        #-----------------------------------------BACKGROUND-----------------------------------
        if event == "_BKG_FOLDER_":
            if os.path.exists(values["_BKG_FOLDER_"]):
                bkg_folder = values["_BKG_FOLDER_"]
                update_config()
            else:
                sg.popup_annoying('Dir not valid')
                
        if event == "_NOISE_":
            if os.path.exists(values["_BKG_FOLDER_"]):
                try:
                    Pedestal(bkg_folder)
                    sg.popup('BKG ready to use')
                    if values['_BKG_FITS_'] == True:
                        try:
                            Rm_Fits_Files(bkg_folder)
                            sg.popup('BKG have been removed')
                        except Exception as e:
                            sg.popup(f"Cannot remove fits : {e}")
                except Exception as e:
                    sg.popup('An ERROR OCCURRED: cannot launch pedesta: {e}')
            else:
                sg.popup_annoying('Dir not found')

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

        #-----------------------------------------SIGNAL-----------------------------------
        if event == "_BKG_FOLDER_A_":
            if os.path.exists(values["_BKG_FOLDER_A_"]):
                bkg_folder_a = values["_BKG_FOLDER_A_"]
        
        if event == "_CORE_":
            if (values['_CORE_'] in ('0123456789')): #Rebin number
                nCore = values["_CORE_"]
            else:
                sg.popup("Only digit allowed")

        if event == "_FIT_FOLDER_": #Folder with .FIT files
            if os.path.exists(values["_FIT_FOLDER_"]):
                fit_folder = values["_FIT_FOLDER_"]

        if event == "_REBIN_":
            if (values['_REBIN_']in ('0123456789')): #Rebin number
                xyRebin = values["_REBIN_"]
            else:
                sg.popup("Only idigit allowed")

        #-----------------------------------------DBSCAN OPTIONS-----------------------------------        
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
        
        if event == "_SIGMA_":
            if (values['_SIGMA_'] in ('0123456789')):   #Number of sigma used for cuts
                sigma = values["_SIGMA_"]
            else:
                sg.popup("Only idigit allowed")
        
        if values['_CLUSTERING_']==False:
            NoClustering = False

        if values['_EVENTS_']==False:
            NoEvent = False
        else:
            NoEvent = True
             
        if values['_RAW_']==True:
            Raw = True

        #-----------------------------------------ANALYSIS-----------------------------------
        if event == "_AN_START_":
            try:
                fit_folder
                try:
                    bkg_folder_a
                    try:
                        Analyze(fit_folder, bkg_folder_a, nCore, xyRebin, sigma, cluster, NoClustering, NoEvent, Raw, Eps)
                        if values['_SIGNAL_FIT_'] == True:
                            try:
                                Rm_Fits_Files(fit_folder)
                            except Exception as e:
                                sg.popup(f"Cannot Remove FITS: {e}")
                                print("Error in Rm_Fits_Files:" , e)
                    except Exception as e:
                        sg.popup(f"Cannot launch Analyze : {e}")
                        print("Error in Analyze:" , e)
                except NameError:
                    sg.popup('Location of bkg is not defined.')
            except NameError:
                sg.popup('Location of data is not defined.')

        #------------------------------------------------------------------------------
        #-------------------------------------CAMERA-----------------------------------

        if event == "_BKG_FOLDER_2_":
            bkg_folder_b = values["_BKG_FOLDER_2_"]
            update_config()
        
        if event == "_BUNCH_":
            bunch = values ["_BUNCH_"]
            update_config()
            
        if event == "_CAMERA_UPDATE_":
            try:
                update_info()
            except Exception:
                sg.popup('Info cant be updated: most likely camera libraries are missing or corrupted')
        
        if event == "_CAPTURE_AND_ANALYZE_PARALLEL_":
            try:
                CaptureAndAnalyze2(StoreDataIn, int(SampleSize),int(WBR),int(WBB),int(exposure),int(gain),bkg_folder_b, xyRebin2, sigma2, cluster2, NoClustering2, NoEvent2, Raw2, Eps2, int(num) , int(length),int(bunch))
            except Exception as e:
                sg.popup(f"Cannot launch CaptureAndAnalyze2: {e}")
                print("Error running CaptureAndAnalyze2:" , e)
        
        if event == "_CLUSTER2_":
            cluster2 = int(values["_CLUSTER2_"])
            update_config()
    
        if event == "_CLUSTERING2_":
            NoClustering2 = values["_CLUSTERING2_"]
            update_config()
        
        if event == "_COLLECT_DATA_":
            try:
                DataChunk(StoreDataIn, file_name, int(SampleSize),int(WBR),int(WBB),int(exposure),int(gain))
            except Exception as e:
                sg.popup(f"Cannot launch DataChunck : {e}")
                print("Error in DataChunck:" , e)

        if event == "_DATA_FOLDER_":
            StoreDataIn=values["_DATA_FOLDER_"]
            update_config()
            
        if event == "_EXPOSURE_":
            exposure=values["_EXPOSURE_"]
            update_config()
            
        if event == "_EPS2_":
            Eps2 = values["_EPS2_"]
            update_config()
            
        if event == "_EVENTS2_":
            NoEvent2 = values["_EVENTS2_"]
            update_config()
            
        if event == "_FILE_NAME_":
            file_name = values["_FILE_NAME_"]
            update_config()
            
        if event == "_FITS_FOLDER_":
            fitin=values["_FITS_FOLDER_"]
            update_config()
            
        if event == "_GAIN_":
            gain=values["_GAIN_"]
            update_config()
        
        if event == "_LEN_":
            length = values["_LEN_"]
            update_config()
        
        if event == "_NOISE2_":
            if os.path.exists(values["_DATA_FOLDER_"]):
                try:
                    Pedestal2(StoreDataIn, int(SampleSize), int(gain), int(WBB), int(WBR), int(exposure))
                    sg.popup('BKG ready to use')
                except Exception as e:
                    sg.popup('An ERROR OCCURRED - cannot launch Pedestal2: {e}')
                    print("Error in Pedestal2: " , e)
            else:
                sg.popup('Dir not found')
        
        if event == "_NUM_":
            num = values["_NUM_"]
            update_config()
            
        if event == "_RAW2_":
            Raw2 = values["_RAW2_"]
            update_config()
           
        if event == "_REBIN2_":
            xyRebin2 = values["_REBIN2_"]
            update_config()
            
        if event == "_SAMPLE_SIZE_":
            SampleSize=values["_SAMPLE_SIZE_"]
            update_config()
        
        if event == "_SIGMA2_":
            sigma2 = values["_SIGMA2_"]
            update_config()
    
        if event == "_WB_B_":
            WBB=values["_WB_B_"]
            update_config()    
            
        if event == "_WB_R_":
            WBR = values["_WB_R_"]
            update_config()

        update_config()
    window.close()
