import glob
import math
import matplotlib
import numpy as np
import FreeSimpleGUI as sg
import sys
sys.path.insert(0, '../../libs')
import utils_v2 as al
import zwoasi as asi
from astropy.io import fits
from matplotlib import use as use_agg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')
from tqdm import tqdm
from tqdm.gui import  tqdm_gui


def bg_map_rt(outMeanPed_file, outStdPed_file, sample_size,GAIN,WB_B,WB_R,EXPO ,hist_pixel=None):
    try:
        camera_id = 0
        camera = asi.Camera(camera_id)
    except Exception as e:
        sg.popup(f" there are trobles: {e}")
    try:
        # Force any single exposure to be halted
        camera.stop_video_capture()
        camera.stop_exposure()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        pass


    if hist_pixel != None:
        print('plotting histogram for pixel:', hist_pixel[0], " ", hist_pixel[1])
        plot_pixel_dist(f, [hist_pixel[0],
                            hist_pixel[1]])  # if hist_pixel differnt form null, plot the histo of that pixel and return
        return
    ny = 4144
    nx = 2822
    # array somma (ogni pixel contine la somma... )
    allSum = np.zeros((nx, ny), dtype=np.int16)
    # array somma^2 (ogni pixel sum(x_i^2)... )
    allSum2 = np.zeros((nx, ny), dtype=np.int16)
    custom_style = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    layout = [
        [sg.Text('Progresso:', size=(10, 1)), sg.ProgressBar(sample_size, orientation='h', size=(20, 20), key='progress')],
        
    ]

    # Crea la finestra
    window = sg.Window('Barra di Progresso', layout, finalize=True)

    try:
        # Use minimum USB bandwidth permitted
        # camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, camera.get_controls()['BandWidth']['MaxValue'])
        camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 95)
        camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, True)
        # Set some sensible defaults. They will need adjusting depending upon
        camera.disable_dark_subtract()
        camera.set_control_value(asi.ASI_GAMMA, 50)
        camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
        camera.set_control_value(asi.ASI_FLIP, 0)
        camera.set_control_value(asi.ASI_GAIN, GAIN)
        camera.set_control_value(asi.ASI_WB_B, WB_B)
        camera.set_control_value(asi.ASI_WB_R, WB_R)
        camera.set_control_value(asi.ASI_EXPOSURE, EXPO)
        camera.set_image_type(asi.ASI_IMG_RAW16)

        # timeout raccomandato
        timeout = (camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 10500
        camera.default_timeout = timeout

        n=0
        for n in tqdm(range(sample_size),desc="Processing", bar_format=custom_style):
            if n == 0:
                camera.start_video_capture()
            n = n + 1.
            window['progress'].update(n)

            image_data = np.empty((2822, 4144), dtype=np.uint16)
            image_data =camera.capture_video_frame()
            image_data = image_data/4
            allSum = allSum + image_data
            allSum2 = allSum2 + image_data ** 2
    finally:
        # Arresta l'esposizione e rilascia la telecamera
        camera.stop_exposure()
        camera.close()

    # mean pedestal for each pixel
    mean = allSum / n
    # pedestal standard deviation:
    std = (allSum2 / n - mean ** 2) ** 0.5

    # write image w mean pedestal
    print('creating pedestal files:\n')
    print('means = ', outMeanPed_file)
    print('rms = ', outStdPed_file)

    al.write_fitsImage(mean, outMeanPed_file, overwrite='True')
    al.write_fitsImage(std, outStdPed_file, overwrite='True')


def plot_pixel_dist(file_list, pixel):
    myVal = []
    for image_file in file_list:
        image_data = al.read_image(image_file) / 4.
        myVal.append(image_data[pixel[0]][pixel[1]])
        # print("val = ",image_data[pixel[0]][pixel[1]])
    npVal = np.array(myVal)
    al.isto_all(npVal)