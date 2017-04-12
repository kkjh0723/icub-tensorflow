/*
 * The Main Controller for testing the network online
 * Written by Jungsik
 * 
 * Update Logs:
 * 2015.05.11: Tested YARP Connection.
 * 2015.06.26:
 * - Start modifying the code for grasp2 (2 object +  human gesture recognition)
 */
#include <yarp/os/all.h>
//#include <yarp/os/Network.h>
//#include <yarp/os/Time.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/sig/Vector.h>
#include <yarp/sig/all.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <string>

#include <stdlib.h> 
#include <math.h> 


#include "controller.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>



using namespace std;
using namespace yarp::os;
using namespace yarp::dev;
using namespace yarp::sig;


void softmaxTransform(double *recvData, double *targetAngles_beforeGraspingTransform);

void readHomePosition(int TYPE, int numJoints, Vector &targetHomeCmd);
double uniformRand(double begin, double end);


string remotePorts_rArm="/icubSim";
string remotePorts_lArm="/icubSim";
string remotePorts_torso="/icubSim";
string remotePorts_head="/icubSim";


// Local ports which sends the data to the robot (or simulator)
string localPort_rArm="/controller/right_arm:o";
string localPort_lArm="/controller/left_arm:o";
string localPort_torso="/controller/torso:o";
string localPort_head="/controller/head:o";

int gZoomLevel;




int main(int argc, char *argv[]) 
{
	srand((unsigned)time(0));
    Network yarp;
    if(!yarp.checkNetwork())
    {
		cout << "YARP server is not available!" <<"\n";
		return -1;
	}

	
	remotePorts_rArm+="/right_arm";
	remotePorts_lArm+="/left_arm";
	remotePorts_torso+="/torso";
	remotePorts_head+="/head";	    
    

	// Connect the local port to the remote port
	Property options_lArm;
	Property options_rArm;
	Property options_torso;
	Property options_head;

	options_rArm.put("device", "remote_controlboard");
	options_rArm.put("local", localPort_rArm.c_str()); //local port names
	options_rArm.put("remote", remotePorts_rArm.c_str()); //where we connect to

	options_lArm.put("device", "remote_controlboard");
	options_lArm.put("local", localPort_lArm.c_str()); //local port names
	options_lArm.put("remote", remotePorts_lArm.c_str()); //where we connect to

	options_torso.put("device", "remote_controlboard");
	options_torso.put("local", localPort_torso.c_str()); //local port names
	options_torso.put("remote", remotePorts_torso.c_str()); //where we connect to

	options_head.put("device", "remote_controlboard");
	options_head.put("local", localPort_head.c_str()); //local port names
	options_head.put("remote", remotePorts_head.c_str()); //where we connect to    
    
	// create a device
	PolyDriver robotDevice_rArm(options_rArm);
	PolyDriver robotDevice_lArm(options_lArm);
	PolyDriver robotDevice_torso(options_torso);
	PolyDriver robotDevice_head(options_head);
	//if (!robotDevice.isValid()) 
	if (!(robotDevice_rArm.isValid() && robotDevice_lArm.isValid() && robotDevice_torso.isValid() && robotDevice_head.isValid()) ) 
	{
		printf("Device not available. Here are the known devices:\n");
		printf("%s", Drivers::factory().toString().c_str());
		return 0;
	}
	
	IPositionControl *pos_rArm;
	IPositionControl *pos_lArm;
	IPositionControl *pos_torso;
	IPositionControl *pos_head;
	
	IEncoders *encs_rArm;
	IEncoders *encs_lArm;
	IEncoders *encs_torso;
	IEncoders *encs_head;

	bool ok;
	ok = robotDevice_rArm.view(pos_rArm);
	ok = ok && robotDevice_rArm.view(encs_rArm);
	ok = ok && robotDevice_lArm.view(pos_lArm);
	ok = ok && robotDevice_lArm.view(encs_lArm);
	ok = ok && robotDevice_torso.view(encs_torso);
	ok = ok && robotDevice_torso.view(pos_torso);
	ok = ok && robotDevice_head.view(pos_head);
	ok = ok && robotDevice_head.view(encs_head);

	if (!ok) 
	{
		printf("Problems acquiring interfaces\n");
		return 0;
	}    

	int nj_rArm=0;
	int nj_lArm=0;
	int nj_torso=0; 
	int nj_head=0; 

	pos_rArm->getAxes(&nj_rArm);
	pos_lArm->getAxes(&nj_lArm);
	pos_torso->getAxes(&nj_torso);
	pos_head->getAxes(&nj_head);
	cout << "\n====================================================\n";
	cout << "[Right Arm]\tNumber of Joints: " << nj_rArm << "\n"; 
	cout << "[Left Arm]\tNumber of Joints: " << nj_lArm << "\n"; 
	cout << "[Torso]\tNumber of Joints: " << nj_torso << "\n"; 
	cout << "[Head]\tNumber of Joints: " << nj_head << "\n"; 
	
	Vector encoders_rArm;
	Vector encoders_lArm;
	Vector encoders_torso;
	Vector encoders_head;

	// command contains the target joint values
	Vector command_rArm;
	Vector command_lArm;
	Vector command_torso;
	Vector command_head;
	
	Vector tmp_rArm;
	Vector tmp_lArm;
	Vector tmp_torso;
	Vector tmp_head;
	
	encoders_rArm.resize(nj_rArm);
	encoders_lArm.resize(nj_lArm);
	encoders_torso.resize(nj_torso);
	encoders_head.resize(nj_head);

	tmp_rArm.resize(nj_rArm);
	tmp_lArm.resize(nj_lArm);
	tmp_torso.resize(nj_torso);
	tmp_head.resize(nj_head);

	command_rArm.resize(nj_rArm);
	command_lArm.resize(nj_lArm);
	command_torso.resize(nj_torso);
	command_head.resize(nj_head);

	cout << "====================================================\n";
	cout << "Setting the reference acceleration and speed ..."; 
	int i;
	for (i = 0; i < nj_rArm; i++) 
	{
		tmp_rArm[i] = JNT_ACC;
	}
	pos_rArm->setRefAccelerations(tmp_rArm.data());

	for (i = 0; i < nj_rArm; i++) 
	{
		tmp_rArm[i] = JNT_SPD;
		pos_rArm->setRefSpeed(i, tmp_rArm[i]);
	}

	for (i = 0; i < nj_lArm; i++) 
	{
		tmp_lArm[i] = JNT_ACC;
	}
	pos_lArm->setRefAccelerations(tmp_lArm.data());

	for (i = 0; i < nj_lArm; i++) 
	{
		tmp_lArm[i] = JNT_SPD;
		pos_lArm->setRefSpeed(i, tmp_lArm[i]);
	}

	for (i = 0; i < nj_torso; i++) 
	{
		tmp_torso[i] = JNT_ACC;
	}
	pos_torso->setRefAccelerations(tmp_torso.data());

	for (i = 0; i < nj_torso; i++) 
	{
		tmp_torso[i] = JNT_SPD;
		pos_torso->setRefSpeed(i, tmp_torso[i]);
	}

	for (i = 0; i < nj_head; i++) 
	{
		tmp_head[i] = JNT_ACC;
	}
	pos_head->setRefAccelerations(tmp_head.data());

	for (i = 0; i < nj_head; i++) 
	{
		tmp_head[i] = JNT_SPD;
		pos_head->setRefSpeed(i, tmp_head[i]);
	}

	cout << "...Done!\n"; 
	//pos->setRefSpeeds(tmp.data())) // originally commented out

	//first read all encoders
	printf("waiting for encoders...\n");
	while(!encs_rArm->getEncoders(encoders_rArm.data()))
	{
		Time::delay(0.1);
		printf(".");
	}
	printf("...Right Arm Done\n");

	while(!encs_lArm->getEncoders(encoders_lArm.data()))
	{
		Time::delay(0.1);
		printf(".");
	}
	printf("...Left Arm Done\n");

	while(!encs_torso->getEncoders(encoders_torso.data()))
	{
		Time::delay(0.1);
		printf(".");
	}
	printf("...Torso Done\n");
	while(!encs_head->getEncoders(encoders_head.data()))
	{
		Time::delay(0.1);
		printf(".");
	}
	printf("...Head Done\n");

	command_rArm=encoders_rArm;
	command_lArm=encoders_lArm;
	command_torso=encoders_torso;
	command_head=encoders_head;
	

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "====================================================\n";
	cout << "Setting up the YARP connection...\n";
	
	BufferedPort<Bottle> port;
	port.open("/controller/softmaxTarget:i"); // This port receives softmax target from the network.
	yarp.connect("/network/softmaxTarget:o","/controller/softmaxTarget:i");
	
	RpcClient port_calcForward;
	port_calcForward.open("/controller/calcForward:rpcClient"); // This port receives softmax target from the network.
	yarp.connect("/controller/calcForward:rpcClient","/network/calcForward:rpcServer");
	
	RpcClient port_captureImage;
	port_captureImage.open("/controller/captureImage:rpcClient");
	yarp.connect("/controller/captureImage:rpcClient","/vision/captureImage:rpcServer");
	
	RpcClient port_worldManip;
	port_worldManip.open("/controller/worldManip");
	yarp.connect("/controller/worldManip","/worldManipulator/targetNum");

	RpcClient port_grasper;
	port_grasper.open("/controller/grasper");
	yarp.connect("/controller/grasper","/fingerGrasper/graspingLevel");

	RpcClient port_world;
	port_world.open("/controller/worldForObj");
	yarp.connect("/controller/worldForObj","/icubSim/world");

	yarp.connect("/vision/image:rpcClient","/network/image:rpcServer");
	
	
	RpcClient port_writeScreen;
	port_writeScreen.open("/controller/writeScreen_targetNum");
	yarp.connect("/controller/writeScreen_targetNum","/screenWriter/targetVideoNum");	
	
#ifdef SAVE_WORLD_VIEW
	// Vision port for getting the world view
    BufferedPort<ImageOf<PixelRgb> > imagePort_world;
    imagePort_world.open("/controller/world");
    Network::connect("/icubSim/cam","/controller/world"); //connect icub sim. to port(need to be modified to real iCub cam.)
#endif
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "====================================================\n";


	/// READ TRIAL LIST
	cout << "====================================================\n";
	cout << "Reading the trial configuration...\n";
	char trialList[256]; // Vision Data
	sprintf(trialList, "./../trialConf/trialList.txt");
	FILE *trialFP;
	trialFP = fopen(trialList, "r");
	int numberOfTrials;
	int res = fscanf(trialFP, "%d", &numberOfTrials); // Read the 1st line, and save it in the variable numberOfSeq
	
	int trials[numberOfTrials][7]; 
	
	for(int idxNumOfSeq = 0 ; idxNumOfSeq < numberOfTrials ; idxNumOfSeq++)
	{
		for(int idxDim = 0 ; idxDim < 7 ; idxDim ++)
		{
			fscanf(trialFP, "%d", &trials[idxNumOfSeq][idxDim]);
			cout << trials[idxNumOfSeq][idxDim] <<"\t";
		}
		cout << "\n";
	}
	fclose(trialFP);

	system("mkdir ./../result/temp");
	cout << "Press Enter to proceed:";
	getchar();


	for(int idxTrial = 0; idxTrial <numberOfTrials ;idxTrial++ )
	{
		int obj1st = trials[idxTrial][TARGET_1ST];
		int other1st = trials[idxTrial][OTHER_1ST];
		int gesture1st = trials[idxTrial][GESTURE_1ST];
		int obj2nd = trials[idxTrial][TARGET_2ND];
		int other2nd = trials[idxTrial][OTHER_2ND];
		int gesture2nd = trials[idxTrial][GESTURE_2ND];
		int classNum = trials[idxTrial][CLASSNUM];
        
        int typeTarget = (obj1st - ((int)obj1st%1000)) / 1000;
        int typeDummy = (other1st - ((int)other1st%1000)) / 1000;
		
		/// Delete all object in the simulation
		Bottle cmd_deleteAllObject;
		Bottle response_cmd_deleteAllObject;
		cmd_deleteAllObject.addDouble(0);
		cmd_deleteAllObject.addDouble(0);
		port_worldManip.write(cmd_deleteAllObject,response_cmd_deleteAllObject); //wait until we get the response

		cout << "====================================================\n";
		cout << "Setting the robot to the home position...\n";
		
		readHomePosition(RIGHT_ARM,nj_rArm,command_rArm);
		readHomePosition(LEFT_ARM,nj_lArm,command_lArm);
		readHomePosition(TORSO,nj_torso,command_torso);
		readHomePosition(HEAD,nj_head,command_head);
	  
		pos_rArm->positionMove(command_rArm.data());
		pos_lArm->positionMove(command_lArm.data());
		pos_torso->positionMove(command_torso.data());
		pos_head->positionMove(command_head.data());

		bool done=false;
		bool done2=false;
		bool done3=false;
		bool done4=false;
		
		while(! (done && done2 && done3 && done4))
		{
			pos_rArm->checkMotionDone(&done);
			pos_lArm->checkMotionDone(&done2);
			pos_torso->checkMotionDone(&done3);
			pos_head->checkMotionDone(&done4);
			Time::delay(0.01);
			fprintf(stderr, ".");
		}
		cout << "Done\n";

		int targetNum = 0;


		char labelName[256]; // motor Teaching Signal
		sprintf(labelName, "./../result/outputMotor_%04d_%04d_%04d_%04d_%04d_%04d.txt",obj1st,other1st,gesture1st,obj2nd,other2nd,gesture2nd);
		char labelName_obj[256]; // motor Teaching Signal
		sprintf(labelName_obj, "./../result/outputObj_%04d_%04d_%04d_%04d_%04d_%04d.txt",obj1st,other1st,gesture1st,obj2nd,other2nd,gesture2nd);


		//Tell the network about the current config
		Bottle cmd2net_config;
		Bottle responsefromnet_config;
		cmd2net_config.addDouble(obj1st);
		cmd2net_config.addDouble(other1st);
		cmd2net_config.addDouble(gesture1st);
		cmd2net_config.addDouble(obj2nd);
		cmd2net_config.addDouble(other2nd);
		cmd2net_config.addDouble(gesture2nd);
		cmd2net_config.addDouble(classNum);
		port_calcForward.write(cmd2net_config,responsefromnet_config); //wait until we get the response
		

		int step = 0;
		while(1)
		{
			cout << "[ " << idxTrial+1 << " / " << numberOfTrials << " ]\t" << classNum << "\t";
			cout << "Step: \t" << step << "\n";
			Time::delay(DELAY_CONTROLLER);

			if(step ==0)
			{
				Bottle cmd_targetNum;
				Bottle response_targetNum;
				cmd_targetNum.addDouble(obj1st);
				cmd_targetNum.addDouble(other1st);
				port_worldManip.write(cmd_targetNum,response_targetNum); //wait until we get the response
				Time::delay(0.1);
				cout << "Done\n";
			}

			Bottle objCmd0;
			Bottle objResp0;
			objCmd0.addString("world");
			objCmd0.addString("get");
            if(typeTarget == 1 || typeTarget == 4)
                objCmd0.addString("sph");
            else if(typeTarget == 2 || typeTarget == 5)
                objCmd0.addString("box");
            else if(typeTarget == 3 || typeTarget == 6)
                objCmd0.addString("cyl");
			objCmd0.addInt(1);
			port_world.write(objCmd0,objResp0); //wait until we get the response
			ofstream SaveFile_obj(labelName_obj, ios::app); 
			SaveFile_obj.precision(6);
			SaveFile_obj.fixed;
SaveFile_obj << objResp0.toString().c_str() << "\n";			
//SaveFile_obj << objResp0.toString().c_str() << "\t";
 /* 		
			objCmd0.clear();
			objResp0.clear();
			objCmd0.addString("world");
			objCmd0.addString("get");
            if(typeDummy == 1 || typeDummy == 4)
                objCmd0.addString("sph");
            else if(typeDummy == 2 || typeDummy == 5)
                objCmd0.addString("box");
            else if(typeDummy == 3 || typeDummy == 6)
                objCmd0.addString("cyl");
			if(typeDummy == typeTarget)
				objCmd0.addInt(2);
			else
				objCmd0.addInt(1);
			port_world.write(objCmd0,objResp0); //wait until we get the response
			
			SaveFile_obj.precision(6);
			SaveFile_obj.fixed;
SaveFile_obj << objResp0.toString().c_str() << "\n";			            
*/            
            
			SaveFile_obj.close();

			if(step < 41)
			{
				Bottle cmd_screenWriter;
				Bottle rsp_screenWriter;
				cmd_screenWriter.addDouble(gesture1st); // Index of Human Gesture Video
				cmd_screenWriter.addDouble(step);
				port_writeScreen.write(cmd_screenWriter,rsp_screenWriter); //wait until we get the response	
        Time::delay(0.01);
			}
			else if(step == 41)
			{
				Bottle cmd_screenWriter;
				Bottle rsp_screenWriter;
				cmd_screenWriter.addDouble(0); // Index of Human Gesture Video
				cmd_screenWriter.addDouble(0);
				port_writeScreen.write(cmd_screenWriter,rsp_screenWriter); //wait until we get the response	
        Time::delay(0.01);
			}

#ifdef SAVE_WORLD_VIEW		
		ImageOf<PixelRgb> *img_rgb_world = imagePort_world.read();
		if(img_rgb_world != NULL) 
		{
			//cout << "... World View loaded successfully.\n";
			//printf("We got an image of size %dx%d\n", img_rgb_world->width(), img_rgb_world->height());
			cv::Mat_<cv::Vec3b> image(img_rgb_world->height(), img_rgb_world->width(), cv::Vec3b(255,255,255));
			for (int y=0; y<img_rgb_world->height(); y++) 
			{
				for (int x=0; x<img_rgb_world->width(); x++) 
				{
					PixelRgb& pixel = img_rgb_world->pixel(x,y);
					image(y,x) = cv::Vec3b(pixel.b, pixel.g, pixel.r);
				}
			}
			vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
			compression_params.push_back(100);
			char str[250];
			sprintf(str, "./../result/outputVisionWorld_%04d_%04d_%04d_%04d_%04d_%04d_%03d.jpg",obj1st,other1st,gesture1st,obj2nd,other2nd,gesture2nd,step);
			cv::imwrite(str, image,compression_params);
		}
#endif		
			
			Bottle fileDesc;
			Bottle fileDescResp;
			fileDesc.addDouble(obj1st);
			fileDesc.addDouble(other1st);
			fileDesc.addDouble(gesture1st);
			fileDesc.addDouble(obj2nd);
			fileDesc.addDouble(other2nd);
			fileDesc.addDouble(gesture2nd);
			
			port_captureImage.write(fileDesc,fileDescResp); //wait until we get the response
			
			
			// At initial step, start the system by capturing image
			if(port_captureImage.getOutputCount() == 0)
			{
				fprintf(stderr,"Vision Module NOT connected... Check the connection\n");
			}
			else
			{
				Bottle cmd;
				if(step == 0)
				{
					cmd.addDouble(1);  // At the initial step, we need specify the zoom (attention) level.
				}
				else
				{
					if(gZoomLevel < 1)
						gZoomLevel = 1;
					if(gZoomLevel > 20)
						gZoomLevel = 20;
						
					//	
					gZoomLevel = 1; // No zoom for VRPGP
					#ifdef ZOOM
						cmd.addDouble(gZoomLevel);
					#else
						cmd.addDouble(1);
					#endif
					cmd.addDouble(step);
					//cmd.addDouble(step%20+1);
				}
					
				//fprintf(stderr,"Ask for Capture ....",cmd.toString().c_str());
				Bottle response;
				port_captureImage.write(cmd,response); //wait until we get the response
				//fprintf(stderr,"%s\n",response.toString().c_str());
			}
			Time::delay(DELAY_CONTROLLER);
			

			if(step >= MOTOR_LENGTH)
			{
				Bottle cmdInit;
				Bottle respInit;
				cmdInit.addDouble(INIT_NET);
				//fprintf(stderr, "Ask the network to initialze the net...");
				port_calcForward.write(cmdInit,respInit); //wait until we get the response
				//fprintf(stderr, "...Done.\n");
				
				break;
			}


			// This is to trigger the first step forward calculation.
			if(port_calcForward.getOutputCount() == 0)
			{
				fprintf(stderr,"Network Module NOT connected... Check the connection\n");
			}
			else
			{
				Bottle cmdCalc;
				cmdCalc.addDouble(CALC);
				//fprintf(stderr,"Ask for Calculation",cmdCalc.toString().c_str());
				Bottle responseCalc;
				port_calcForward.write(cmdCalc,responseCalc); //wait until we get the response
				//fprintf(stderr,"%s\n",responseCalc.toString().c_str());
			}
			Time::delay(DELAY_CONTROLLER);
			
			Bottle *input = port.read();
			if (input!=NULL) 
			{

				double recvData[input->size()]; // input->size() should be TOTAL_SOFTMAX_DIM.
				double targetAngles_beforeGraspingTransform[NUM_SIG_DIM]; // grasping is represented as the value 0~10; Attention: 1~20
				double targetAngles_finger[NUM_JOINT_FINGERS];
				if(input->size() != TOTAL_SOFTMAX_DIM)
				{
					cout << "\nProblem while reciving the data from the network to the controller!\n";
					cout << "network data size: "<< input->size() << "\tcontroller data size: "<<TOTAL_SOFTMAX_DIM<<endl;
					return 0; //exit(1);
					
				}
				for (int i=0; i<input->size(); i++) 
				{
					recvData[i] = input->get(i).asDouble();
				}
				
				//cout << "Data Received. Softmax Transform started...\n";
				softmaxTransform(recvData,targetAngles_beforeGraspingTransform);
				//cout << "ZOOM\n" << targetAngles_beforeGraspingTransform[NUM_SIG_DIM-1] << "\t" << (int)targetAngles_beforeGraspingTransform[NUM_SIG_DIM-1];
				gZoomLevel = (int)targetAngles_beforeGraspingTransform[IDX_ZOOM_LEVEL];
				

				ofstream SaveFile_zoom(labelName, ios::app); 
				SaveFile_zoom.precision(6);
				SaveFile_zoom.fixed;
				for (int i=0; i<NUM_SIG_DIM; i++) 
				{
					SaveFile_zoom << targetAngles_beforeGraspingTransform[i] << "\t";
				}
				SaveFile_zoom << "\n";
				SaveFile_zoom.close(); 
				

				// To ensure other joints of the head is not moving from the home position.
				//readHomePosition(HEAD,nj_head,command_head);
				command_head[0] = targetAngles_beforeGraspingTransform[0];
				command_head[2] = targetAngles_beforeGraspingTransform[1];
				
				
				for(int idxJoint = 0 ; 	idxJoint < NUM_JOINT_ARM ; idxJoint++)
				{
					//command_rArm[idxJoint] = targetAngles_beforeGraspingTransform[idxJoint+2];
					command_lArm[idxJoint] = targetAngles_beforeGraspingTransform[idxJoint+2];
				}
				//bool ret22_rArm=encs_rArm->getEncoders(encoders_rArm.data());
				bool ret22_lArm=encs_lArm->getEncoders(encoders_lArm.data());
				for(int idxJoint = NUM_JOINT_ARM; idxJoint < nj_lArm ; idxJoint++)
				{
					//command_rArm[idxJoint] = encoders_rArm[idxJoint];
					command_lArm[idxJoint] = encoders_lArm[idxJoint];
				}
				
				if(step >0)//3)
				{
					pos_head->positionMove(command_head.data());
					//pos_rArm->positionMove(command_rArm.data());						
					pos_lArm->positionMove(command_lArm.data());						
				}
				
				if(step < 109 || step> 120)//Before&After gRasping)
				{
					done=false;
					done3=false;
					int tries = 0;
					while(! (done && done3))
					{
						//pos_rArm->checkMotionDone(&done);
						pos_lArm->checkMotionDone(&done);
						pos_head->checkMotionDone(&done3);
						Time::delay(0.01);
						tries++;
						if(tries > 100)
							break;
					}
				}

				if(targetAngles_beforeGraspingTransform[IDX_GRASPING_LEVEL] >= 2)
        {
        Bottle graspingLevel;
				Bottle graspResp;
				graspingLevel.addDouble(targetAngles_beforeGraspingTransform[IDX_GRASPING_LEVEL]);
				port_grasper.write(graspingLevel,graspResp);
        }

				
				/*
				Time::delay(0.01);			
				bool ret_rArm=encs_rArm->getEncoders(encoders_rArm.data());
				bool ret_head=encs_head->getEncoders(encoders_head.data());

				if (!(ret_rArm&&ret_head))
					fprintf(stderr, "Error receiving encoders, check connectivity with the robot\n");
				*/
				/*else
				{
					// Verbose
					cout.precision(0); cout<<fixed; 					
					cout << "Target(Encoder):" << command_head[0] << "(" << encoders_head[0] << ") "
						<< command_head[2] << "(" << encoders_head[2] << ")\t";
					for(int idxJointTemp = 0 ; idxJointTemp < NUM_JOINT_ARM ; idxJointTemp++)
					{
						cout << command_rArm[idxJointTemp] << "(" << encoders_rArm[idxJointTemp] <<") ";
					}
					cout << "\t";
					cout << targetAngles_beforeGraspingTransform[IDX_GRASPING_LEVEL] << "\t" << gZoomLevel <<"\n";
				}
				*/ 
			}
			step+=STEP_INTERVAL;
		}	
		system("mv ./../result/output* ./../result/temp	");
	}
	robotDevice_rArm.close();
	robotDevice_lArm.close();
	robotDevice_torso.close();
	robotDevice_head.close();

    Network::disconnect("/network/softmaxTarget:o","/controller/softmaxTarget:i");
    Network::disconnect("/controller/nextStepReady:o","/network/nextStepReady:i");
    return 0;
}


void softmaxTransform(double *recvData, double *targetAngles_beforeGraspingTransform)
{
	double dimMinMax[NUM_SIG_DIM][2];
	char labelName_minMax[256];
	sprintf(labelName_minMax, "./../softmaxConfig/dimMinMaxFile.txt");
	FILE *targetFP_minMax;
	targetFP_minMax = fopen(labelName_minMax, "r");
	for(int tempIdx = 0; tempIdx < NUM_SIG_DIM ; tempIdx++)
	{
		fscanf(targetFP_minMax, "%lf", &dimMinMax[tempIdx][0]);
		fscanf(targetFP_minMax, "%lf", &dimMinMax[tempIdx][1]);
	}
	if(targetFP_minMax)
	{
		fclose(targetFP_minMax);
		targetFP_minMax = NULL;
	}
/*
-55.0000	41.0000
-18.0700	39.5800
-78.8200	24.8100
-15.0000	118.0000
-28.0000	90.9000
-15.0000	120.9500
-15.0000	74.3700
-31.3200	15.0500
-15.0000	55.0200
-15.0000	25.0000



*/	
	
	int trueMax[11];
	int trueMin[11];
	trueMin[0] = -40;
	trueMax[0] = 30;//30;
	
	trueMin[1] = -55;
	trueMax[1] = 55;
	
	// ARM
	trueMin[2] = -95;
	trueMax[2] = 10;
	
	trueMin[3] = 0;
	trueMax[3] = 160;
	
	trueMin[4] = -36;
	trueMax[4] = 79;
	
	trueMin[5] = 15;
	trueMax[5] = 105;
	
	trueMin[6] = -90;
	trueMax[6] = 90;
	
	trueMin[7] = -90;
	trueMax[7] = 0;
	
	trueMin[8] = -19;
	trueMax[8] = 39;
	
	trueMin[9] = 0;
	trueMax[9] = 60;
	
//	trueMin[10] = 0;
//	trueMax[10] = 20;
	
	
	for (int idxSigDim = 0 ; idxSigDim < NUM_SIG_DIM ; idxSigDim++)
	{
		double xMin = dimMinMax[idxSigDim][0];
		double xMax = dimMinMax[idxSigDim][1];
		int precision = 100;
		
		double x[X_LENGTH];
		char labelName[256]; 
		sprintf(labelName, "./../softmaxConfig/x_dim%d.txt", idxSigDim+1);
		FILE *targetFP;
		targetFP = fopen(labelName, "r");
		for(int tempIdx = 0; tempIdx < X_LENGTH ; tempIdx++)
		{
			fscanf(targetFP, "%lf", &x[tempIdx]);
		}
		if(targetFP)
		{
			fclose(targetFP);
			targetFP = NULL;
		}
		
		double invX[INVX_LENGTH];
		char labelName2[256];
		sprintf(labelName2, "./../softmaxConfig/invX_dim%d.txt", idxSigDim+1);
		FILE *targetFP2;
		targetFP2 = fopen(labelName2, "r");
		for(int tempIdx = 0; tempIdx < INVX_LENGTH ; tempIdx++)
		{
			fscanf(targetFP2, "%lf", &invX[tempIdx]);
		}
		if(targetFP2)
		{
			fclose(targetFP2);
			targetFP2 = NULL;
		}
		
		

		int idxTempDim = 0;
		double valuesToBeTransformed[SOFTMAX_DIMENSION];
		double tempSum = 0;
		for (int idxDim=idxSigDim*10 ; idxDim < idxSigDim*10+10 ; idxDim++)
		{
			valuesToBeTransformed[idxTempDim] = recvData[idxDim];
			tempSum = tempSum + valuesToBeTransformed[idxTempDim];
			idxTempDim++;
		}
		
		double estimatedValue = 0;
		for (int idxDim = 0 ; idxDim < SOFTMAX_DIMENSION ; idxDim++)
		{
			estimatedValue = estimatedValue + ((invX[idxDim] + invX[idxDim+1])/2)*valuesToBeTransformed[idxDim];
		}
		


		if(estimatedValue > trueMax[idxSigDim])
			estimatedValue = trueMax[idxSigDim];
		if(estimatedValue < trueMin[idxSigDim])
			estimatedValue = trueMin[idxSigDim];
		
		
		targetAngles_beforeGraspingTransform[idxSigDim] = estimatedValue;
	}
}



void readHomePosition(int TYPE, int numJoints, Vector &targetHomeCmd)
{
	char labelName2[256];
	switch(TYPE)
	{
		case RIGHT_ARM:
			sprintf(labelName2, "./../homePositions/home_rArm.txt");
			break;
		case LEFT_ARM:
			sprintf(labelName2, "./../homePositions/home_lArm.txt");
			break;
		case TORSO:
			sprintf(labelName2, "./../homePositions/home_torso.txt");
			break;
		case HEAD:
			sprintf(labelName2, "./../homePositions/home_head.txt");
			break;
	}

	double targetHome[numJoints];
	FILE *targetFP2;
	targetFP2 = fopen(labelName2, "r");
	for(int tempIdx = 0; tempIdx < numJoints ; tempIdx++)
	{
		fscanf(targetFP2, "%lf", &targetHome[tempIdx]);
		targetHomeCmd[tempIdx] = targetHome[tempIdx];
	}
	if(targetFP2)
	{
		fclose(targetFP2);
		targetFP2 = NULL;
	}
	
	
}


double uniformRand(double begin, double end) 
{
  double number = rand() / (RAND_MAX + 1.0) * (end - begin) + begin;
  return number;
}
