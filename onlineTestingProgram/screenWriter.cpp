/*
 * Screen Writer 
 * by Jungsik
 * 
 * This program displays images on the screen in the iCub simulation
 * To enable the screen, please change iCub_Activation.ini in the simConfig folder under $YARP
 * INPUT RECEIVED: VIDEO INDEX & STEP (2 DOUBLE NUMBERS)
 * RETURN: VIDEO INDEX & STEP
 * 
 * Update Logs:
 * 2015.06.22. Displaying on the screen succeed.
 * 2015.06.23. When the program receives two 0s, (0 0) then it displays the empty screen.
 * 2015.06.26. Human Gesture Video Index Added. 
 * 3 digits: HUMAN_INDEX	TYPE(1 (indicating left) OR 2(indicating right))	TRIAL (0 TO 9)
 * 
 */
#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include "screenWriter.h"

using namespace cv;
using namespace std;
using namespace yarp::os;
using namespace yarp::sig;
int main(int argc, char *argv[]) 
{
    Network yarp;

    RpcServer port; // **** PORT - SERVER
    port.open("/screenWriter/targetVideoNum"); // To receive index & steps of human gesture videos

    Port imagePort;
    imagePort.open("/screenWriter/imageOut");
    Network::connect("/screenWriter/imageOut","/icubSim/texture/screen"); //connect icub sim. screen

    while(1)
    {
		Bottle cmd;
		Bottle response;
		int videoIndex;
		int step;
		port.read(cmd, true); // Server waiting for the input
		videoIndex = (int)cmd.get(0).asDouble();
		step = (int)cmd.get(1).asDouble();
		cout << "Video Index: " << videoIndex <<"\tStep: " << step <<"\n";
		
		char imgFileName[255];
		if( videoIndex ==0 && step ==0)
			sprintf(imgFileName, "../../data/gestureVideos/grayScreen.jpg");			
			//sprintf(imgFileName, "./gestureVideos/temp.jpeg");			
		else
		{
			if(step == 0 || step>HUMAN_GESTURE_LENGTH)
			{
				//sprintf(imgFileName, "./gestureVideos/video_%03d_%03d.jpeg",videoIndex,HUMAN_GESTURE_LENGTH-1);			
				sprintf(imgFileName, "../../data/gestureVideos/grayScreen.jpg");	
			}		
			else
			{
				sprintf(imgFileName, "../../data/gestureVideos/checkerboard.jpg");			
				//sprintf(imgFileName, "../../../../gestureVideos/video_%03d_%03d.jpeg",videoIndex,step);			
			}
		}
		Mat image;
		image = imread(imgFileName, CV_LOAD_IMAGE_COLOR);   // Read the file			
		ImageOf<PixelBgr> yarpImage;
		yarpImage.setExternal(image.data,image.size[1],image.size[0]);
		
    imagePort.write(yarpImage);			
    if(step > HUMAN_GESTURE_LENGTH + 10)
  		Time::delay(0.01);
    else
      Time::delay(0.1);
		port.reply(cmd);
	}
    return 0;
}
