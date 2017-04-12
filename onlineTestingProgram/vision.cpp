/*
 * Vision Module
 * input:
 * -captureImage : receives from the controller. capture image when captureImage = true. * 
 * output:
 * -image: sends image to the network. RPC. so wait until the network receives the image.
 * 
 * NOTE)
 * For training. it might be required to have functionality for saving image!
 * 
 * Important Update Logs:
 * 2015.05.12 Jungsik
 * - YARP connection tested. (RPC)
 * 2015.05.14 Jinhyung
 * - Get Image from iCub and process
 * 2015.05.26 Jungsik
 * - Getting Zoom/No ZOOM from vision added
 * 2015.06.30
 * - Modified for grasp2
 */
#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include "controller.h"




using namespace std;
using namespace yarp::os;
using namespace yarp::sig;
int main(int argc, char *argv[]) 
{
    Network yarp;
    
    RpcClient port;
    port.open("/vision/image:rpcClient"); // This port sends image to the network
    
    RpcServer port_captureImage;
    port_captureImage.open("/vision/captureImage:rpcServer");
    
    
    BufferedPort<ImageOf<PixelRgb> > imagePort;
    imagePort.open("/vision/rawImage");
    Network::connect("/icubSim/cam/left","/vision/rawImage"); //connect icub sim. to port(need to be modified to real iCub cam.)
	
    ////////////////////////////////////////////////////////////////////////////////////////////

    //int step = 0;
	while(1)
	{
		Bottle fileDesc;
		Bottle fileDescResp;
		port_captureImage.read(fileDesc,true);
		double obj1st = fileDesc.get(0).asDouble();
		double other1st = fileDesc.get(1).asDouble();
		double gesture1st = fileDesc.get(2).asDouble();
		double obj2nd = fileDesc.get(3).asDouble();
		double other2nd = fileDesc.get(4).asDouble();
		double gesture2nd = fileDesc.get(5).asDouble();

		char labelName[256];
		//sprintf(labelName, "./result/outputVision.txt_xPos_%lf_yPos_%lf_rot_%d_obs_%lf",xPos,yPos,(int)rot,xObs);
		sprintf(labelName, "./result/outputVision.txt_%04d_%04d_%04d_%04d_%04d_%04d",(int)obj1st,(int)other1st,(int)gesture1st,(int)obj2nd,(int)other2nd,(int)gesture2nd);
		port_captureImage.reply(fileDescResp);
		


		//fprintf(stderr,"Step: %d \n Waiting for capture signal...\t",step);
		Bottle cmd;
		Bottle response;
		port_captureImage.read(cmd,true);
		//double captureNow = cmd.get(0).asDouble();
		int captureNow = (int)cmd.get(0).asDouble();
		int step = (int)cmd.get(1).asDouble();
		
		if(captureNow<1)
			captureNow = 1;
		if(captureNow>20)
			captureNow = 20;
		
		//fprintf(stderr,"capture? = %.0f\t",captureNow);
		if(captureNow) //Write image to the network
		{
			if(port.getOutputCount() == 0)
			{
				fprintf(stderr,"NOT connected... Check the connection\n");
			}
			else
			{
				
				//read image from cam.				
				ImageOf<PixelRgb> *img_rgb = imagePort.read();  //read image from port
				if(img_rgb != NULL) 
				{
					//cout << "... iCub Camera image loaded successfully.\n";

					IplImage *img_Ipl = cvCreateImage(cvSize(img_rgb->width(), img_rgb->height()), IPL_DEPTH_8U, 1); 
					IplImage *img_gray_Ipl = cvCreateImage(cvSize(IMG_COL, IMG_ROW), IPL_DEPTH_8U, 1); 					
					cvCvtColor(img_rgb->getIplImage(), img_Ipl, CV_RGB2GRAY); //RGB 2 Gray 
					//if(captureNow>1)cvShowImage("RGB",img_rgb->getIplImage());
					
					//zoom//
					int c_width = 8;
					int c_height = 6;
					CvRect rect;
					rect.x = (c_width/2)*(captureNow-1);
					rect.y = (c_height/2)*(captureNow-1);
					rect.height = img_rgb->height() - c_height*(captureNow-1);
					rect.width = img_rgb->width() - c_width*(captureNow-1);				
					cvSetImageROI(img_Ipl,rect);
					
					//IplImage *img_Ipl_cropped = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
					//cvCopy(img_Ipl,img_Ipl_cropped);
					
					cvNamedWindow("Attention",0);// 2nd argument '0' is for fixed window
					cvResizeWindow("Attention",320,240);
					//if(captureNow>1)cvShowImage("Attention",img_Ipl);
					////////
					
					cvResize(img_Ipl, img_gray_Ipl, CV_INTER_LINEAR);//resize
					
					//Original image//
					cvResetImageROI(img_Ipl);
					//if(captureNow>1) cvShowImage("Original",img_Ipl);
					//if(captureNow>1) cvWaitKey(0);
					////////

					// Sending image to the network

					ofstream SaveFile(labelName, ios::app); 
					SaveFile.precision(6);
					SaveFile.fixed;

					Bottle cmd2net;
					unsigned char uc_pix;
					double d_pix;

					if(0)//step == 0)
					{
						for (int idxPixel = 0 ; idxPixel < (IMG_ROW * IMG_COL) ; idxPixel++)
						{
							cmd2net.addDouble(0);
							SaveFile << 0 << "\t";
						}
					}
					else
					{
						for (int idxPixel = 0 ; idxPixel < (IMG_ROW * IMG_COL) ; idxPixel++)
						{
							uc_pix = img_gray_Ipl->imageData[idxPixel];
							d_pix = (double)uc_pix;
						#ifdef USE_DATASET_VISION
							cmd2net.addDouble(image[step][idxPixel]);
						#else
							cmd2net.addDouble((d_pix/255-0.5)*2);
						#endif
						
							SaveFile << (d_pix/255-0.5)*2 << "\t";
						}
					}
					SaveFile << "\n";
					SaveFile.close();
					//fprintf(stderr,"Send image to the network ..",cmd2net.toString().c_str());
					Bottle responseNet;
					port.write(cmd2net,responseNet);
					//fprintf(stderr,"%s\n",responseNet.toString().c_str());
					
					cvReleaseImage(&img_gray_Ipl);//release IplImage(not sure necessary)				
				}
				
			}
			Time::delay(DELAY_VISION);
			response.addString("NetworkReceivedImage!");
			step+=STEP_INTERVAL;
		}
		else
		{
			Time::delay(DELAY_VISION);
			response.addString("Network_DO_NOT_ReceivedImage!");
		}
		port_captureImage.reply(response); // Response to the controller
		
	}
    return 0;
}
