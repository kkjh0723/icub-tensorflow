#include <stdlib.h>
#include <float.h>
#include "cnn.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <yarp/os/all.h>
#include "controller.h"

#include <time.h> // To check the running time.

using namespace std;
using namespace yarp::os;

int main(int argc, char* argv[])
{
    // YARP SETTING
    Network yarp;
    BufferedPort<Bottle> port;
    port.open("/network/softmaxTarget:o"); // This port receives softmax target from the network.
    
    //BufferedPort<Bottle> port_nextStepReady;
    RpcServer port_calcForward;
    port_calcForward.open("/network/calcForward:rpcServer"); // This port receives from the controller
    
    //BufferedPort<Bottle> port_img;
    RpcServer port_img;
    port_img.open("/network/image:rpcServer"); // This port receives softmax target from the network.


while(1)
{

	double netOut[TOTAL_SOFTMAX_DIM];
	for(int idxJoint = 0; idxJoint < TOTAL_SOFTMAX_DIM ; idxJoint++)
	{
		netOut[idxJoint] = 0.0;
	}
	cout << "...target loaded successfully.\n";
	cout << "================================================\n";
	cout << "Please locate netInform in ./result directory and make sure that you have checked backup & configuration.\n";
	int idxConfigFile = 1; // not USED.

	//set random seed
	srand(SEED_NUMBER);							
	CCNN cnn;
	//0227 test
	int *stepSize_temp;
	int tempBatchSize = 1;
	stepSize_temp = new int[tempBatchSize];
	stepSize_temp[0] = 2; // we maintain only 2 steps of data!

	cnn.AllocWRTCal(1, stepSize_temp, 1);

	cnn.LoadWeights("./result/netInform");





	//receive config info from controller
	Bottle cmdConfig;
	Bottle responseConfig;
	port_calcForward.read(cmdConfig,true); // waiting until receiving image.
	int obj1st = (int)cmdConfig.get(0).asDouble();	
	int other1st = (int)cmdConfig.get(1).asDouble();	
	int gesture1st = (int)cmdConfig.get(2).asDouble();	
	int obj2nd = (int)cmdConfig.get(3).asDouble();	
	int other2nd = (int)cmdConfig.get(4).asDouble();	
	int gesture2nd = (int)cmdConfig.get(5).asDouble();	
	int classNum = (int)cmdConfig.get(6).asDouble();	

	
	cnn.setClassNumber(classNum);	
	
	responseConfig.addDouble(0);
	port_calcForward.reply(responseConfig);
	


	
//while(1)
//{
    int step = 0;
    while (1)//(step < MOTOR_LENGTH)
    {
		// Check image input.
		double recvImg[(IMG_ROW * IMG_COL)];
		//cout << "global step\t" << step << "\n";
		//fprintf(stderr,"Waiting for image input...");
		Bottle cmd;
		Bottle response;
		port_img.read(cmd,true); // waiting until receiving image.
		if(cmd.size() != (IMG_ROW * IMG_COL))
		{
			cout << "\nProblem while reciving the 'input_img' from the vision module!\n";
			//exit(1);
			return 0;
		}
		else
		{
			//fprintf(stderr,"image received.\n");
            // Now we have image here
            for (int idxPixel=0; idxPixel<cmd.size(); idxPixel++) 
            {
                recvImg[idxPixel] = cmd.get(idxPixel).asDouble();
            }
			response.addString("...received");
			port_img.reply(response);
		}
		
		
		// Check for calculation
		//fprintf(stderr,"Waiting for calculation...");
		Bottle cmdCalc;
		Bottle responseCalc;
		port_calcForward.read(cmdCalc,true); // waiting until receiving image.
		if(cmdCalc.size() != 1)
		{
			cout << "\nProblem while reciving the 'cmdCalc' from the controller module!\n";
			//exit(1);
			return 0;
		}
		double calcNow = cmdCalc.get(0).asDouble();
		//fprintf(stderr,"calc? = %.0f\t",calcNow);
		
		// FORWARD CALC.
		if(calcNow == 2)
		{
			fprintf(stderr, "Start initializing the network...");
			cnn.DeleteWRTCal(1, stepSize_temp);
			Bottle initNetResp;
			initNetResp.addDouble(0);
			port_calcForward.reply(initNetResp);
			fprintf(stderr, "...done \n");
			break;
		}
		else if(calcNow == CALC)//if(calcNow)
		{
			//fprintf(stderr,"here. \n");
			
			cnn.Calculate_online(recvImg,netOut,step,(double)obj1st,(double)other1st,(double)gesture1st,(double)classNum);
			//fprintf(stderr,"calc done. \n");

			Bottle& output = port.prepare();
			output.clear();
			for (int idxSM = 0 ; idxSM < TOTAL_SOFTMAX_DIM ; idxSM++)
			{
				//output.addDouble(target[step][idxSM]);
				output.addDouble(netOut[idxSM]);
			}
			//cout << output.toString().c_str() << "\n";
			port.write();
			//fprintf(stderr,"output sent to the controller\n");
			
			responseCalc.addString("...Calc Done");
		}
		else
		{
			fprintf(stderr,"calc not done. \n");
			responseCalc.addString("...Calc NOT Done");
		}
		Time::delay(DELAY_NETWORK);
		
		////////////////////////////////////////////////////////////////
		// END OF FORWARD DYNAMICS
		////////////////////////////////////////////////////////////////
		port_calcForward.reply(responseCalc);
		step+=STEP_INTERVAL;
		if(0)//if(step >= TEACHING_LENGTH)
		{
			cnn.DeleteWRTCal(1, stepSize_temp);
			break;
		}

    }
    
}
////////////////////////////////////////////////////////////////////////////////////////////////
	return 0;
}
