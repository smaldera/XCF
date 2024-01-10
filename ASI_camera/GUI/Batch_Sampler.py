import numpy as np
import zwoasi as asi
from astropy.io import fits
import time

def capture(camera,file_name, file_path, sample_size, WB_R, WB_B, EXPO, GAIN):
    try:
        # Use minimum USB bandwidth permitted
        #camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, camera.get_controls()['BandWidth']['MinValue'])

        # Set some sensible defaults. They will need adjusting depending upon
        # the sensitivity, lens and lighting conditions used.
        camera.disable_dark_subtract()
        camera.set_control_value(asi.ASI_GAMMA, 50)
        camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
        camera.set_control_value(asi.ASI_FLIP, 0)
        camera.set_control_value(asi.ASI_GAIN, GAIN)
        camera.set_control_value(asi.ASI_WB_B, WB_B)
        camera.set_control_value(asi.ASI_WB_R, WB_R)
        camera.set_control_value(asi.ASI_EXPOSURE, EXPO)
        camera.set_image_type(asi.ASI_IMG_RAW16)

        try:
            # Force any single exposure to be halted
            camera.stop_video_capture()
            camera.stop_exposure()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            pass

        for i in range (sample_size):

            # Ottieni i dati dell'immagine
            data = np.empty((2822, 4144), dtype=np.uint16)
            data = camera.capture()

            # Crea l'header FITS
            header = fits.Header()
            header['EXPTIME'] = EXPO
            header['DATE-OBS'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())

            # Salva l'immagine come file FITS
            hdu = fits.PrimaryHDU(data, header=header)
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(file_path + "/"+file_name+str(i)+".FIT", overwrite=True)

        print(f"Immagine salvata in: {file_path}")

    finally:
        # Arresta l'esposizione e rilascia la telecamera
        camera.stop_exposure()
        camera.close()

