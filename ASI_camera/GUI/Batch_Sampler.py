import numpy as np
import PySimpleGUI as sg
import time
import zwoasi as asi
from astropy.io import fits
from tqdm import  tqdm

#THIS FUNCTION WILL BE MODIFIED AS IT'S JUST A COPY OF PEDESTAL
def capture(file_name, file_path, sample_size, WB_R, WB_B, EXPO, GAIN):
    try:
        camera_id = 0
        camera = asi.Camera(camera_id)
    except Exception as e:
        sg.popup(f"Can't connect to camera: {e}")
    try:
        # Force any single exposure to be halted
        camera.stop_video_capture()
        camera.stop_exposure()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        pass
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

