import glob
import math
import matplotlib
import numpy as np
import FreeSimpleGUI as sg
import sys
sys.path.insert(0, '../../libs')
import time
import utils_v2 as al
import zwoasi as asi

from astropy.io import fits
from matplotlib import use as use_agg
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from tqdm import tqdm
from tqdm.gui import  tqdm_gui


def bg_map(bg_shots_path, outMeanPed_file, outStdPed_file, hist_pixel=None):
    ny = 4144
    nx = 2822
    allSum = np.zeros((nx, ny), dtype=np.int16) # Array made of sums of each pixel
    allSum2 = np.zeros((nx, ny), dtype=np.int16) # Array made of squared sum of each pixel
    
    # Images file list:
    f = glob.glob(bg_shots_path + "/*.FIT")
    len_file = len(f)
    print("Pedestals from :", bg_shots_path)
    
    if hist_pixel != None:
        # If hist_pixel differnt form null, plot the histo of that pixel and return
        print('Plotting histogram for pixel: ', hist_pixel[0], " , ", hist_pixel[1])
        plot_pixel_dist(f, [hist_pixel[0], hist_pixel[1]])  
        return
    
    # SimpleGui style for progress bar
    custom_style = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    layout = [
        [sg.Text('Working:', size=(10, 1)), sg.ProgressBar(len_file, orientation='h', size=(20, 20), key='progress')],
    ]
    window = sg.Window('Barra di Progresso', layout, finalize=True)

    n = 0.
    for image_file in tqdm(f,desc="Processing", bar_format=custom_style):
        image_data = al.read_image(image_file)/4.
        allSum = allSum + image_data
        allSum2 = allSum2 + image_data**2
        n += 1.
        window['progress'].update(n)
    mean = allSum / n # Mean pedestal for each pixel
    std = (allSum2 / n - mean ** 2) ** 0.5 # Pedestal standard deviation

    # Write image w mean pedestal
    print('creating pedestal files:\n')
    print('means = ', outMeanPed_file)
    print('rms = ', outStdPed_file)
    al.write_fitsImage(mean, outMeanPed_file, overwrite='True')
    al.write_fitsImage(std, outStdPed_file, overwrite='True')
    

def bg_map_rt(bg_shots_path, outMeanPed_file, outStdPed_file, sample_size, GAIN, WB_B, WB_R, EXPO, hist_pixel=None):
    ny = 4144
    nx = 2822
    allSum = np.zeros((nx, ny), dtype=np.int16) # Array made of sums of each pixel
    allSum2 = np.zeros((nx, ny), dtype=np.int16) # Array made of squared sum of each pixel
    
    camera = checkup()
    
    # Images file list:
    f = glob.glob(bg_shots_path + "/*.FIT")
    print("Pedestals from :", bg_shots_path)

    if hist_pixel != None:
        print('Plotting histogram for pixel:', hist_pixel[0], ", ", hist_pixel[1])
        plot_pixel_dist(f, [hist_pixel[0],
                            hist_pixel[1]])  # if hist_pixel differnt form null, plot the histo of that pixel and return
        return
    
    # SimpleGui style for progress bar
    custom_style = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    layout = [
        [sg.Text('Progresso:', size=(10, 1)), sg.ProgressBar(sample_size, orientation='h', size=(20, 20), key='progress')],
    ]
    window = sg.Window('Barra di Progresso', layout, finalize=True)

    try:
        camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 95)
        camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, True)
        # Set some sensible defaults. They will need adjusting depending upon needs
        camera.disable_dark_subtract()
        camera.set_control_value(asi.ASI_GAMMA, 50)
        camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
        camera.set_control_value(asi.ASI_FLIP, 0)
        camera.set_control_value(asi.ASI_GAIN, GAIN)
        camera.set_control_value(asi.ASI_WB_B, WB_B)
        camera.set_control_value(asi.ASI_WB_R, WB_R)
        camera.set_control_value(asi.ASI_EXPOSURE, EXPO)
        camera.set_image_type(asi.ASI_IMG_RAW16)
        # timeout needed
        timeout = (camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 10500
        camera.default_timeout = timeout

        n=0
        for n in tqdm(range(sample_size),desc="Processing", bar_format=custom_style):
            if n == 0:
                camera.start_video_capture()
            n = n + 1.
            window['progress'].update(n)
            image_data = np.empty((nx, ny), dtype=np.uint16)
            image_data =camera.capture_video_frame()
            image_data = image_data/4
            allSum = allSum + image_data
            allSum2 = allSum2 + image_data ** 2
    finally:
        # Stops exposure and closes the camera
        camera.stop_exposure()
        camera.close()
    mean = allSum / n # Mean pedestal for each pixel
    std = (allSum2 / n - mean ** 2) ** 0.5 # Pedestal standard deviation

    # Write image w mean pedestal
    print('creating pedestal files:\n')
    print('means = ', outMeanPed_file)
    print('rms = ', outStdPed_file)
    al.write_fitsImage(mean, outMeanPed_file, overwrite='True')
    al.write_fitsImage(std, outStdPed_file, overwrite='True')


def capture(file_name, file_path, sample_size, WB_R, WB_B, EXPO, GAIN):
    camera = checkup()
    try:
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
        timeout = (camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 10500
        camera.default_timeout = timeout

        layout_capture = [
            [sg.Text('Cattura in corso:', size=(15, 1)), sg.ProgressBar(sample_size, orientation='h', size=(20, 20), key='progress_capture')],
        ]
        window_capture = sg.Window('Cattura in corso', layout_capture, finalize=True)
        progress_bar_capture = window_capture['progress_capture']

        for i in tqdm(range (sample_size)):
            if i == 0:
                camera.start_video_capture()
            progress_bar_capture.UpdateBar(i)
            data = np.empty((2822, 4144), dtype=np.uint16)
            data = camera.capture_video_frame()
            # FITS header
            header = fits.Header()
            header['EXPTIME'] = EXPO
            header['DATE-OBS'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
            # Saving Image.FITS
            hdu = fits.PrimaryHDU(data, header=header)
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(file_path + "/"+file_name+str(i)+".FIT", overwrite=True)
        print(f"Saved in: {file_path}")
    finally:
        camera.stop_exposure()
        camera.close()
        window_capture.Close()


def checkup():
    try:
        camera_id = 0
        camera = asi.Camera(camera_id)
    except Exception as e:
        sg.popup(f"Camera not found in bg_map_rt: {e}")
    try:
        # Force any single exposure to be halted
        camera.stop_video_capture()
        camera.stop_exposure()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        pass
    return camera
    
    
def plot_pixel_dist(file_list, pixel):
    myVal = []
    for image_file in file_list:
        image_data = al.read_image(image_file) / 4.
        myVal.append(image_data[pixel[0]][pixel[1]])
    npVal = np.array(myVal)
    al.isto_all(npVal)