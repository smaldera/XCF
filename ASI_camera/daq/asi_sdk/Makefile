ver =debug
platform = x64

CC = g++
#INCLIB = /usr/local/include
#LDLIB = /usr/local/lib
#OPENCV = -lopencv_core -lopencv_highgui -lopencv_imgproc#$(shell pkg-config --cflags opencv) $(shell pkg-config --libs opencv) -I/usr/include/opencv2
OPENCV = -lopencv_core -lopencv_highgui -lopencv_imgproc#$(shell pkg-config --cflags opencv) $(shell pkg-config --libs opencv) -I/usr/include/opencv4/opencv2

#USB =  -I../libusb/include  -L../libusb/$(platform) -lusb-1.0  


#ASIROOTDIR=/home/maldera/Desktop/eXTP/softwareXCF/XCF/ASI_camera/daq/asi_sdk
ASIROOTDIR=.

LIBSPATH = -L $(ASIROOTDIR)/lib/$(platform) -I$(ASIROOTDIR)/include



ifeq ($(ver), debug)
DEFS = -D_LIN -D_DEBUG 
CFLAGS = -g  -I $(INCLIB) -L $(LDLIB) $(DEFS) $(COMMON) $(LIBSPATH)  -lpthread  -DGLIBC_20   -Wl,-rpath=$(ASIROOTDIR)/lib/x64
else
DEFS = -D_LIN
CFLAGS =  -O3 -I $(INCLIB) -L $(LDLIB) $(DEFS) $(COMMON) $(LIBSPATH)  -lpthread  -DGLIBC_20    -Wl,-rpath=$(ASIROOTDIR)/lib/x64
endif




CFLAGS += -m64
CFLAGS += -lrt


all:ASItakeShots_simo.cpp 
	$(CC)  ASItakeShots_simo.cpp -o ASItakeShots_simo  $(CFLAGS) -lASICamera2 -lcfitsio

all:ASItakeShots_simo_v2.cpp
	$(CC)  ASItakeShots_simo_v2.cpp -o ASItakeShots_simo_v2  $(CFLAGS) -lASICamera2 -lcfitsio
	 

clean:
	rm -f ASItakeShots_simo

#pkg-config libusb-1.0 --cflags --libs
#pkg-config opencv --cflags --libs

