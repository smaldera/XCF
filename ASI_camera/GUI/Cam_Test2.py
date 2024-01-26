import numpy as np
import zwoasi as asi
from astropy.io import fits
import time

def capture_as_fit():
   
    file_path='/home/xcf/testCMOS_genn2024/'

    try:
        cameraID = 0
        camera = asi.Camera(cameraID)
        
        # Configura la telecamera
        
        # Use minimum USB bandwidth permitted
        camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 95)

        # Set some sensible defaults. They will need adjusting depending upon
        # the sensitivity, lens and lighting conditions used.
        camera.disable_dark_subtract()

        camera.set_control_value(asi.ASI_GAIN, 150)
        camera.set_control_value(asi.ASI_WB_B, 99)
        camera.set_control_value(asi.ASI_WB_R, 75)
        camera.set_control_value(asi.ASI_GAMMA, 50)
        camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
        camera.set_control_value(asi.ASI_FLIP, 0)
        camera.set_control_value(asi.ASI_EXPOSURE, 30000)

        camera.set_image_type(asi.ASI_IMG_RAW16)

        try:
            # Force any single exposure to be halted
            camera.stop_video_capture()
            camera.stop_exposure()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            pass

        exposure_time=10000
        # Ottieni i dati dell'immagine
        for i in 1000:
            data = np.empty((2822, 4144), dtype=np.uint16)
            data = camera.capture()

            # Crea l'header FITS
            header = fits.Header()
            header['EXPTIME'] = exposure_time
            header['DATE-OBS'] = time.strftime('%Y-%m-%dT%H:%M:%S_'  + str(i), time.gmtime())

            # Salva l'immagine come file FITS
            hdu = fits.PrimaryHDU(data, header=header)
            hdulist = fits.HDUList([hdu])
            file_name='foto_'+str(i)
            hdulist.writeto(file_path + "/"+file_name+".fits", overwrite=True)

        print(f"Immagine salvata in: {file_path}")

    finally:
        # Arresta l'esposizione e rilascia la telecamera
        camera.stop_exposure()
        camera.close()


def GetCameraID():

    num_cameras = asi.get_num_cameras()
    if num_cameras == 0:
        print('No cameras found')

    cameras_found = asi.list_cameras()  # Models names of the connected cameras

    if num_cameras == 1:
        camera_id = 0
        print('Found one camera: %s' % cameras_found[0])
    else:
        print('Found %d cameras' % num_cameras)
        for n in range(num_cameras):
            print('    %d: %s' % (n, cameras_found[n]))
        # TO DO: allow user to select a camera
        camera_id = 0
        print('Using #%d: %s' % (camera_id, cameras_found[camera_id]))
    return(camera_id)




capture_as_fit()
