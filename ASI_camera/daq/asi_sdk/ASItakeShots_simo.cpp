#include "stdio.h"
//#include "opencv2/highgui/highgui_c.h"
//#include "opencv2/imgproc.hpp"  // aggiunto da me!!!!
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
       //char filename[] = "prova.fit";  
	int bitpix   =  16;         /* 16-bit short signed integer pixel values */
	long naxis    =   2;        /* 2-dimensional image                      */
	long naxes[2] = {width ,height };   /* image is 300 pixels wide by 200 rows */
	int fitsStatus = 0;         /* initialize status before calling fitsio routines */

	int fpixel = 1;                               /* first pixel to write      */
	int nelements = naxes[0] * naxes[1];          /* number of pixels to write */
	


	fits_create_file(&fptr, nomefile, &fitsStatus) ;
//fits_create_img(fptr,  bitpix, naxis, naxes, &fitsStatus);
	fits_create_img(fptr, USHORT_IMG , naxis, naxes, &fitsStatus);
	
	//	fits_write_img(fptr, TSHORT, fpixel, nelements, pImg16bit  , &fitsStatus);
	fits_write_img(fptr, TUSHORT, fpixel, nelements, pImg16bit  , &fitsStatus);

	fits_close_file(fptr, &fitsStatus);            /* close the file */
	//	cout<<" status = "<<fitsStatus<<endl;
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
       
	  
        int width;
	const char* bayer[] = {"RG","BG","GR","GB"};

	int height;
	int i;
	char c;
	
	//int count=0;
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

	cout<<"SIMO: connecting to camera 0... "<<endl;

	if(ASIOpenCamera(0) != ASI_SUCCESS)
	{
		printf("OpenCamera error,are you root?,press any key to exit\n");
		getchar();
		return -1;
	}
	ASIInitCamera(CamNum);

	ASI_ERROR_CODE err = ASIEnableDarkSubtract(CamNum, "dark.bmp");
	if(err == ASI_SUCCESS)
		printf("load dark ok\n");
	else
		printf("load dark failed %d\n", err);

	printf("%s information\n",ASICameraInfo.Name);
	int iMaxWidth, iMaxHeight;
	iMaxWidth = ASICameraInfo.MaxWidth;
	iMaxHeight =  ASICameraInfo.MaxHeight;
	printf("resolution:%dX%d\n", iMaxWidth, iMaxHeight);
	if(ASICameraInfo.IsColorCam)
		printf("Color Camera: bayer pattern:%s\n",bayer[ASICameraInfo.BayerPattern]);
	else
		printf("Mono camera\n");
	
	ASI_CONTROL_CAPS ControlCaps;
	int iNumOfCtrl = 0;
	ASIGetNumOfControls(CamNum, &iNumOfCtrl);
	for( i = 0; i < iNumOfCtrl; i++)
	{
		ASIGetControlCaps(CamNum, i, &ControlCaps);
		printf("%s\n", ControlCaps.Name);
	}



	long ltemp = 0;
	ASI_BOOL bAuto = ASI_FALSE;
	ASIGetControlValue(CamNum, ASI_TEMPERATURE, &ltemp, &bAuto);
	printf("sensor temperature:%02f\n", (float)ltemp/10.0);

        long wb_r=-1;
	long wb_b=-1;
	ASIGetControlValue(CamNum, ASI_WB_R, &wb_r, &bAuto);
	ASIGetControlValue(CamNum, ASI_WB_B, &wb_b, &bAuto);

	cout<<" wb_r="<<wb_r<<" wb_b = "<<wb_b<<endl; 
	
	int bin = 1, Image_type;
	
	width = iMaxWidth;
	height = iMaxHeight;
	bin=1;
	Image_type=2; // raw 16?
	//	int myGain=120;
	
	//while(ASI_SUCCESS != ASISetROIFormat(CamNum, width, height, bin, (ASI_IMG_TYPE)ASI_IMG_RAW16));//IMG_RAW16 ?????
	while(ASI_SUCCESS != ASISetROIFormat(CamNum, width, height, bin, (ASI_IMG_TYPE)ASI_IMG_RAW8));//IMG_RAW16 ?????

	ASIGetROIFormat(CamNum, &width, &height, &bin, (ASI_IMG_TYPE*)&Image_type);
	//printf("\nset image format %d %d %d %d success, start privew, press ESC to stop \n", width, height, bin, Image_type);

	
	
	long imgSize = width*height*(1 + (Image_type==ASI_IMG_RAW16));
       
	unsigned char* imgBuf = new unsigned char[imgSize];

	ASISetControlValue(CamNum, ASI_GAIN, myGain, ASI_FALSE);

	//int exp_ms;
	//exp_ms=833;
	cout<<" exposure (us) = "<<exp_us<<endl;
	//ASISetControlValue(CamNum, ASI_EXPOSURE, exp_ms*1000, ASI_FALSE);
	ASISetControlValue(CamNum, ASI_EXPOSURE, exp_us, ASI_FALSE);
	ASISetControlValue(CamNum, ASI_BANDWIDTHOVERLOAD, 95, ASI_FALSE);
	ASISetControlValue(CamNum, ASI_HIGH_SPEED_MODE, 1, ASI_FALSE);

	long lbw = 0;
	//	ASIGetControlValue(CamNum, ASI_BANDWIDTH, &lbw, &bAuto);
	//printf("bandwd:%02f\n", lXSbw);

	cout<<"bandwidth= "<< ASICameraInfo.BandWidth<<endl;
	
	cout<<" SIMO: ... starting exposures"<<endl;

	ASI_EXPOSURE_STATUS status;
	int n=0;
	
	while (n<n_shots){
	  cout<<" taking shot n. "<<n<<endl;
	  ASIStartExposure(CamNum, ASI_FALSE);
	 
	  //usleep(10000);//1ms
	  status = ASI_EXP_WORKING;
	  while(status == ASI_EXP_WORKING){
			ASIGetExpStatus(CamNum, &status);		
	  }

	  if(status == ASI_EXP_SUCCESS){
		       ASIGetDataAfterExp(CamNum, imgBuf, imgSize);
	  }		

	 
	  
	  unsigned short *pImg16bit = (unsigned short *)imgBuf;

	 
	
	  // salvo immagine in fits file?????
	  int fitsStatus = 0; 
     
	  string filename2=base_path+"shot_"+to_string ( n )+".FIT";
	
	  fitsStatus= write_fits_file(const_cast<char*>(filename2.c_str()), 	pImg16bit,  width,  height  );
	
	  n=n+1;
	  
	
	}
	///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
	
	
	ASIStopExposure(CamNum);
	ASICloseCamera(CamNum);




	





}






