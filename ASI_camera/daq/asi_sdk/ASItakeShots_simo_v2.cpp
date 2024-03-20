#include "stdio.h"
#include "ASICamera2.h"
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "pthread.h"
#include <iostream>
#include "fitsio.h"
#include <typeinfo>
#include <string>
using namespace std;


int write_fits_file(char* nomefile, 	unsigned short *pImg16bit, int width, int height  ){

	// salvo immagine in fits file
	fitsfile *fptr;   
	int bitpix   =  16;         /* 16-bit short signed integer pixel values */
	long naxis    =   2;        /* 2-dimensional image                      */
	long naxes[2] = {width ,height };   /* image is width  pixels wide by height rows */
	int fitsStatus = 0;         /* initialize status before calling fitsio routines */
	int fpixel = 1;                               /* first pixel to write      */
	int nelements = naxes[0] * naxes[1];          /* number of pixels to write */
	
	fits_create_file(&fptr, nomefile, &fitsStatus) ;
	fits_create_img(fptr, USHORT_IMG , naxis, naxes, &fitsStatus);
	fits_write_img(fptr, TUSHORT, fpixel, nelements, pImg16bit  , &fitsStatus);
	fits_close_file(fptr, &fitsStatus);            /* close the file */

 return (fitsStatus);
}


int  main(int argc, char *argv[])
{

        if (argc!=5){
	  cout<<"error:  usage: ./ASItakeShots_simo  exposure gain shots path"<<endl;
	  exit(0);
	}

	int myGain=atoi(argv[2]); 
	int exp_us=atoi(argv[1]); 
	int n_shots=atoi(argv[3]);
	string base_path=argv[4];

	cout<<" exp = "<<exp_us<<endl;
	cout<<" myGain="<<myGain<<endl;
	cout<<" n_shots="<<n_shots<<endl;
       	cout<<" path="<<base_path<<endl;
       
	ASI_BOOL bAuto = ASI_FALSE;
        int width;
	int height;
	int i;
	char c;
	
	int CamNum=0;

	int numDevices = ASIGetNumOfConnectedCameras();
	if(numDevices <= 0)
	{
		printf("no camera connected, press any key to exit\n");
		getchar();
		return -1;
	}
	else
		printf("attached cameras:\n");

	ASI_CAMERA_INFO ASICameraInfo;

	ASIGetCameraProperty(&ASICameraInfo, CamNum);
	printf("%d %s\n",i, ASICameraInfo.Name);

	cout<<"SIMO: connecting to camera "<<CamNum<<"... "<<endl;
	if(ASIOpenCamera(CamNum) != ASI_SUCCESS)
	{
		printf("OpenCamera error,are you root?,press any key to exit\n");
		getchar();
		return -1;
	}
	cout<<"SIMO: inizializing camera "<<CamNum<<"... "<<endl;
	ASIInitCamera(CamNum);

	printf("%s information\n",ASICameraInfo.Name);
	int iMaxWidth, iMaxHeight;
	iMaxWidth = ASICameraInfo.MaxWidth;
	iMaxHeight =  ASICameraInfo.MaxHeight;
	printf("resolution:%dX%d\n", iMaxWidth, iMaxHeight);
		
	ASI_CONTROL_CAPS ControlCaps;
	int iNumOfCtrl = 0;
	ASIGetNumOfControls(CamNum, &iNumOfCtrl);
	for( i = 0; i < iNumOfCtrl; i++)
	{
		ASIGetControlCaps(CamNum, i, &ControlCaps);
		long val;
		ASIGetControlValue(CamNum,ControlCaps.ControlType, &val, &bAuto);
		cout<<ControlCaps.Name<<" "<<val<<endl;
	}



	long ltemp = 0;
	ASIGetControlValue(CamNum, ASI_TEMPERATURE, &ltemp, &bAuto);
	printf("sensor temperature:%02f\n", (float)ltemp/10.0);

        	
	int bin = 1;
	
	width = iMaxWidth;
	height = iMaxHeight;
	bin=1;
	//int Image_type=2; // raw 16?
	
	
	while(ASI_SUCCESS != ASISetROIFormat(CamNum, width, height, bin, (ASI_IMG_TYPE)ASI_IMG_RAW16));
	

	long lBuffSize= height*width*2;
	unsigned char *pBuffer= (unsigned char *) malloc(lBuffSize) ;
	
	
	ASISetControlValue(CamNum, ASI_GAIN, myGain, ASI_FALSE);
	
	cout<<" exposure (us) = "<<exp_us<<endl;
	ASISetControlValue(CamNum, ASI_EXPOSURE, exp_us, ASI_FALSE);
	ASISetControlValue(CamNum, ASI_BANDWIDTHOVERLOAD, 95, ASI_FALSE);
	ASISetControlValue(CamNum, ASI_HIGH_SPEED_MODE, 1, ASI_FALSE);

	
	
	cout<<" SIMO: ... start video capture"<<endl;
	
	ASIStartVideoCapture(CamNum);
	
	ASI_EXPOSURE_STATUS status;
	int n=0;
		
	while (n<n_shots){
	           ASI_ERROR_CODE exposurestatus= ASIGetVideoData(CamNum, pBuffer, lBuffSize, exp_us/1000.);		 
		  if (exposurestatus==ASI_SUCCESS){

		    cout<<" taking shot n. "<<n<<endl;
		    unsigned short *pImg16bit = (unsigned short *)pBuffer;
		    int fitsStatus = 0; 
		    string filename2=base_path+"shot_"+to_string ( n )+".FIT";
		    fitsStatus= write_fits_file(const_cast<char*>(filename2.c_str()), 	pImg16bit,  width,  height  );
	  	  
		    n++;
		  }
	 	
	}
	
	
	ASIStopVideoCapture(CamNum);
	ASICloseCamera(CamNum);


}






