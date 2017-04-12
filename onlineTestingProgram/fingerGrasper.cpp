/*
 * The Finger Grasper
 * Written by Jungsik
 * 
 * Update Logs:
 * 2015.06.23: Getting sensor values done. Moving Thumb faster done
 */
#include <yarp/os/all.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/sig/Vector.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

#include "fingerGrasper.h"




using namespace std;
using namespace yarp::os;
using namespace yarp::dev;
using namespace yarp::sig;


void graspingTransform(double graspingLevel, double *targetAngles_finger);
double getCurrentGrasping(void);

string remotePorts_lArm="/icubSim/left_arm";
Port port_sensors;

// Local ports which sends the data to the robot (or simulator)
string localPort_lArm="/fingerGrasper/left_arm:o";


int main(int argc, char *argv[]) 
{
    Network yarp;
    if(!yarp.checkNetwork())
    {
		cout << "YARP server is not available!" <<"\n";
		return -1;
	}

	Property options_lArm;
	options_lArm.put("device", "remote_controlboard");
	options_lArm.put("local", localPort_lArm.c_str()); //local port names
	options_lArm.put("remote", remotePorts_lArm.c_str()); //where we connect to
	PolyDriver robotDevice_lArm(options_lArm);
	if (!(robotDevice_lArm.isValid()) ) 
	{
		printf("Device not available. Here are the known devices:\n");
		printf("%s", Drivers::factory().toString().c_str());
		return 0;
	}
	
	IPositionControl *pos_lArm;
	IControlMode2 		*ictrlLarm;
	IEncoders *encs_lArm;
	bool ok;
	ok = robotDevice_lArm.view(pos_lArm);
	ok = ok && robotDevice_lArm.view(encs_lArm);
	ok = ok && robotDevice_lArm.view(ictrlLarm);
	if (!ok) 
	{
		printf("Problems acquiring interfaces\n");
		return 0;
	}    

	int nj_lArm=0;
	pos_lArm->getAxes(&nj_lArm);
	cout << "\n====================================================\n";
	cout << "[Left Arm]\tNumber of Joints: " << nj_lArm << "\n"; 
	
	Vector encoders_lArm;
	Vector command_lArm;
	Vector tmp_lArm;
	
	encoders_lArm.resize(nj_lArm);
	tmp_lArm.resize(nj_lArm);
	command_lArm.resize(nj_lArm);

	cout << "====================================================\n";
	cout << "Setting the reference acceleration and speed ..."; 
	int i;
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
	cout << "...Done!\n"; 

	//first read all encoders
	printf("waiting for encoders...\n");
	while(!encs_lArm->getEncoders(encoders_lArm.data()))
	{
		Time::delay(0.1);
		fprintf(stderr,".");
	}
	printf("...Left Arm Done\n");


	command_lArm=encoders_lArm;


	RpcServer port;
	port.open("/fingerGrasper/graspingLevel");
	
	//Port port_sensors;
	port_sensors.open("/fingerGrasper/sensorInput");
	yarp.connect("/icubSim/skin/left_hand_comp","/fingerGrasper/sensorInput");


	int gTuningFail = 0;
	while(1)
	{
		Bottle cmd_graspLevel;
		Bottle response;
		double desiredGraspLevel;
		port.read(cmd_graspLevel, true); // RPC Server waiting for the input
		desiredGraspLevel = cmd_graspLevel.get(0).asDouble();
		if(desiredGraspLevel < 2)
			desiredGraspLevel = 1;
		else if(desiredGraspLevel > 9)
			desiredGraspLevel = 10;

		double target[16];			
		double targetAngles_finger[NUM_JOINT_FINGERS];
		
		
		graspingTransform(desiredGraspLevel,targetAngles_finger); // The last element contains grasping level.
		
		Time::delay(0.01);
		while(!encs_lArm->getEncoders(encoders_lArm.data()))
		{
			Time::delay(0.01);
			fprintf(stderr,".");
		}
		for(int idxJoint = 0 ; 	idxJoint < NUM_JOINT_ARM ; idxJoint++)
		{
			command_lArm[idxJoint] = encoders_lArm[idxJoint]; // ARM (7 DOF) should not be changing while grasping!
		}
		
/*
  		for(int jnt=7 ; jnt < 15; jnt++)
		{
			ictrlLarm->setPositionMode(jnt);
		}		
*/

		int tempIdx = 0;
		for(int idxJoint = 7; idxJoint < nj_lArm ; idxJoint++)
		{
			command_lArm[idxJoint] = targetAngles_finger[tempIdx]; // We control the fingers only.
			tempIdx++;
		}
		if(desiredGraspLevel <= 1)
		{
		 	command_lArm[NUM_JOINT_ARM+1] = 10;//20; // THUMB
		 	command_lArm[NUM_JOINT_ARM+4] = 0; // THUMB
		 	command_lArm[NUM_JOINT_ARM+5] = 0; // THUMB
		 	command_lArm[NUM_JOINT_ARM+6] = 0; // THUMB
		 	command_lArm[NUM_JOINT_ARM+7] = 0; // THUMB
		 	command_lArm[NUM_JOINT_ARM+8] = 0; // THUMB
		}

		if(desiredGraspLevel < 10)
			gTuningFail = 0;

		
		//if(1) //used while recording
		if(desiredGraspLevel > 1)
		{
			Bottle input;
			//Time::delay(0.1);
			Time::delay(0.005); //1129
			
			//port_sensors.read(input);
			
			double total_index = 0;
			double total_middle = 0;
			double total_ring = 0;
			double total_little = 0;
			double total_thumb = 0;

			double tempCurrentGrasp = getCurrentGrasping();
			if(desiredGraspLevel == 10 && tempCurrentGrasp > MINIMUM_GRASPING_SENSOR_VALUE)
			{
					/// we already got it. so skip!
					/// HOWEVER, if some of the fingers are not touching the object, try to make them grasp!
					if(total_thumb > 200)
					{	
						command_lArm[8] = encoders_lArm[8];
						command_lArm[9] = encoders_lArm[9];
						command_lArm[10] = encoders_lArm[10];
					}
					if(total_index > 200)
					{	
						command_lArm[11] = encoders_lArm[11];
						command_lArm[12] = encoders_lArm[12];
					}
					if(total_middle > 200)
					{	
						command_lArm[13] = encoders_lArm[13];
						command_lArm[14] = encoders_lArm[14];
					}
					if(total_little > 200 || total_ring > 200) 
					{	
						command_lArm[15] = encoders_lArm[15];
					}
					
					
					pos_lArm->positionMove(command_lArm.data());						
					bool done=false;
					int tempTRY = 0;
					while(!done)
					{
						pos_lArm->checkMotionDone(&done);
						Time::delay(0.01);
						tempTRY++;
						int maxT = 0;
						if(desiredGraspLevel < 9)
							maxT = 30;
						else
							maxT = 100;

						double avg_grasp = getCurrentGrasping();
						if(avg_grasp > MINIMUM_GRASPING_SENSOR_VALUE || tempTRY > maxT)//30
							break;
					}					
			}
			else
			{
				int tempIdx = 0;
				
				if(total_thumb > 200)
				{	
					command_lArm[8] = encoders_lArm[8];
					command_lArm[9] = encoders_lArm[9];
					command_lArm[10] = encoders_lArm[10];
				}
				if(total_index > 200)
				{	
					command_lArm[11] = encoders_lArm[11];
					command_lArm[12] = encoders_lArm[12];
				}
				if(total_middle > 200)
				{	
					command_lArm[13] = encoders_lArm[13];
					command_lArm[14] = encoders_lArm[14];
				}
				if(total_little > 200 || total_ring > 200) 
				{	
					command_lArm[15] = encoders_lArm[15];
				}
				
				
				pos_lArm->positionMove(command_lArm.data());						
				bool done=false;
				int tempTRY = 0;
				while(!done)
				{
					pos_lArm->checkMotionDone(&done);
					Time::delay(0.01);
					tempTRY++;
					int maxT = 0;
					if(desiredGraspLevel < 9)
						maxT = 30;
					else
						maxT = 100;

					double avg_grasp = getCurrentGrasping();

					if(avg_grasp > MINIMUM_GRASPING_SENSOR_VALUE || tempTRY > maxT)//30
						break;
				}
			}			
		}
		
	

		double currentGrasp = getCurrentGrasping();
		// To avoid the case in which the fingers are not fully grasping the object even with the graspingLevel = 10.
		if(desiredGraspLevel == 10 && currentGrasp < MINIMUM_GRASPING_SENSOR_VALUE)
		{
			//cout << "Tunining Started\n";
			int numTry = 1;
			while(currentGrasp<MINIMUM_GRASPING_SENSOR_VALUE)
			{
				if(gTuningFail == 1)
				{
					break;
				}

				Bottle input;
				port_sensors.read(input);
				
				double total_index = 0;
				double total_middle = 0;
				double total_ring = 0;
				double total_little = 0;
				double total_thumb = 0;
				for(int i = 0 ; i < 12 ; i++)
				{
					total_index += input.get(i).asDouble();
					total_middle += input.get(i+12).asDouble();
					total_ring += input.get(i+24).asDouble();
					total_little += input.get(i+36).asDouble();
					total_thumb += input.get(i+48).asDouble();
				}
							
				fprintf(stderr,"%d\tTUNING: %.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",numTry,total_index/12,total_middle/12,total_ring/12,total_little/12,total_thumb/12);
				response.clear();
				double avg_grasp = (total_index/12+total_middle/12+total_ring/12+total_little/12+total_thumb/12)/5;
				
				currentGrasp = avg_grasp;
				double graspIndex = total_index;
				double graspMiddle = total_middle;
				double graspRing = total_ring;
				double graspLittle = total_little;
				double graspThumb = total_thumb;
					
				bool ret_lArm=encs_lArm->getEncoders(encoders_lArm.data());
				if(graspIndex < MINIMUM_GRASPING_SENSOR_VALUE)
				{
					command_lArm[11] = encoders_lArm[11] + 10;
					command_lArm[12] = encoders_lArm[12] + 10;
				}
				if(graspMiddle < MINIMUM_GRASPING_SENSOR_VALUE)
				{
					command_lArm[13] = encoders_lArm[13] + 10;
					command_lArm[14] = encoders_lArm[14] + 10;
				}
				if(graspRing < MINIMUM_GRASPING_SENSOR_VALUE || graspLittle < MINIMUM_GRASPING_SENSOR_VALUE)
				{
					command_lArm[15] = encoders_lArm[15] + 10;
				}
				/*
				if(graspThumb < MINIMUM_GRASPING_SENSOR_VALUE)
				{
					command_lArm[8] = encoders_lArm[8] + 2;
					command_lArm[9] = encoders_lArm[9] + 2;
					command_lArm[10] = encoders_lArm[10] + 2;
				}
				*/ 
				pos_lArm->positionMove(command_lArm.data());						

				bool doneTemp=false;
				int tempTRY = 0;
				while(!doneTemp)
				{
					pos_lArm->checkMotionDone(&doneTemp);
					Time::delay(0.01);
					tempTRY++;
					if(tempTRY > 30)
						doneTemp = true;
				}

			
				numTry++;
				if(numTry > MAX_NUM_TRY)
				{
					gTuningFail = 1;
					break;
				}
				
			}
		}
		
		
		response.addDouble(currentGrasp);		
		//Time::delay(0.01);
		Time::delay(0.001); //1129
		port.reply(response);
	}	
	robotDevice_lArm.close();

    return 0;
}



// 0: Fingers stratight, 1: Grasping --> to finger's real joint values
void graspingTransform(double graspingLevel, double *targetAngles_finger)
{
	// Finger 1,2,3 --> Thumb. Need to Move faster?
	double fingerMinMax[NUM_JOINT_FINGERS][2];
	fingerMinMax[0][0] = FINGER_0_MIN;
	fingerMinMax[0][1] = FINGER_0_MAX;
	fingerMinMax[1][0] = FINGER_1_MIN;
	fingerMinMax[1][1] = FINGER_1_MAX;
	fingerMinMax[2][0] = FINGER_2_MIN;
	fingerMinMax[2][1] = FINGER_2_MAX;
	fingerMinMax[3][0] = FINGER_3_MIN;
	fingerMinMax[3][1] = FINGER_3_MAX;
	fingerMinMax[4][0] = FINGER_4_MIN;
	fingerMinMax[4][1] = FINGER_4_MAX;
	fingerMinMax[5][0] = FINGER_5_MIN;
	fingerMinMax[5][1] = FINGER_5_MAX;
	fingerMinMax[6][0] = FINGER_6_MIN;
	fingerMinMax[6][1] = FINGER_6_MAX;
	fingerMinMax[7][0] = FINGER_7_MIN;
	fingerMinMax[7][1] = FINGER_7_MAX;
	fingerMinMax[8][0] = FINGER_8_MIN;
	fingerMinMax[8][1] = FINGER_8_MAX;
	
	for (int idxFinger = 0 ; idxFinger < NUM_JOINT_FINGERS ; idxFinger++)
	{
		double value = 0;
		double newGraspingLevel = 1;
		if(idxFinger >= 1 && idxFinger <=3) // Finger 1,2,3 --> Thumb. Need to Move faster?
		{
			newGraspingLevel = graspingLevel *2;
			if(newGraspingLevel > 10)
				newGraspingLevel = 10;
		}
		else
		{
			newGraspingLevel = graspingLevel;
		}
		value = (((fingerMinMax[idxFinger][1] - fingerMinMax[idxFinger][0]) * (newGraspingLevel - GRASPING_LEVEL_MIN) ) / (GRASPING_LEVEL_MAX - GRASPING_LEVEL_MIN) ) + fingerMinMax[idxFinger][0];
		targetAngles_finger[idxFinger] = value;
	}
}

double getCurrentGrasping(void)
{
	Bottle input;
	Bottle response;
	port_sensors.read(input);
	
	double total_index = 0;
	double total_middle = 0;
	double total_ring = 0;
	double total_little = 0;
	double total_thumb = 0;
	for(int i = 0 ; i < 12 ; i++)
	{
		total_index += input.get(i).asDouble();
		total_middle += input.get(i+12).asDouble();
		total_ring += input.get(i+24).asDouble();
		total_little += input.get(i+36).asDouble();
		total_thumb += input.get(i+48).asDouble();
	}
				
	//fprintf(stderr,"%d\tTUNING: %.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",numTry,total_index/12,total_middle/12,total_ring/12,total_little/12,total_thumb/12);
	response.clear();
	double avg_grasp = (total_index/12+total_middle/12+total_ring/12+total_little/12+total_thumb/12)/5;
	return avg_grasp;
}
