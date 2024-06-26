import sys
import numpy as np
from matplotlib import pyplot as plt
import glob
sys.path.insert(0, '../../libs')
import utils_v2 as al
from tqdm import tqdm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import PySimpleGUI as sg
matplotlib.use('TkAgg')
import math
from tqdm.gui import  tqdm_gui
from matplotlib import use as use_agg


def plot_pixel_dist(file_list, pixel):
    myVal = []
    for image_file in file_list:
        image_data = al.read_image(image_file) / 4.
        myVal.append(image_data[pixel[0]][pixel[1]])
        # print("val = ",image_data[pixel[0]][pixel[1]])

    npVal = np.array(myVal)
    al.isto_all(npVal)


def bg_map(bg_shots_path, outMeanPed_file, outStdPed_file, ny=4144, nx=2822, draw=1, hist_pixel=None):
    # lista file immagini:
    f = glob.glob(bg_shots_path + "/*.FIT")
    len_fil = len(f)

    print("pedestals from :", bg_shots_path)

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
        [sg.Text('Progresso:', size=(10, 1)), sg.ProgressBar(len_fil, orientation='h', size=(20, 20), key='progress')],
        
    ]

    # Crea la finestra
    window = sg.Window('Barra di Progresso', layout, finalize=True)

    n = 0.
    for image_file in tqdm(f,desc="Processing", bar_format=custom_style):
        n = n + 1.
        window['progress'].update(n)

        # print(n," --> ", image_file)
        # if n%10==0:
        #   frac=float(n/len(f))*100.
        #   print("Pedestal-> processed ",n," files  (  %.2f %%)" %frac )
        image_data = al.read_image(image_file) / 4.
        allSum = allSum + image_data
        allSum2 = allSum2 + image_data ** 2

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


def ShowPlots():
    al.plot_image(mean)
    al.isto_all(mean)

    al.plot_image(std)
    al.isto_all(std)

