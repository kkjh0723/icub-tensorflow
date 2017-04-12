/*
 * The World Manipulator
 * Written by Jungsik and Edited by Naveen
 *
 * Update Logs:
 * 2015.05.19. RPC Connection tested.
 * 2015.06.22. Now we're having 2 objects on the task space
 * For coordinate positions: http://wiki.icub.org/wiki/Simulator_README  see under world section
 * To make the objects, send 2 object numbers to /worldManipulator/targetNum, then it will return object information
 * object number) 1st digit: obj.Pos(X), 1~4, 2nd digit: 1 for portrait, 2 for landscape, 3rd digit: orientation(angle),1~5
 * e.g.) 111 324 --> left object on position1, portrait, orientation 1, right object on position 3, landscape, orientation 4
 * if 000 000 are given from other programs, the world manipulator will delete all objects.
 *
 */

#include <yarp/os/all.h>
//#include <yarp/os/Network.h>
//#include <yarp/os/Time.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/sig/Vector.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

#include "worldManipulator.h"

using namespace std;
using namespace yarp::os;
using namespace yarp::dev;
using namespace yarp::sig;


void getObjPosRot(int seqNum, double *objInfo);
int main(int argc, char *argv[])
{
    Network yarp;
    if(!yarp.checkNetwork())
    {
        cout << "YARP server is not available!" <<"\n";
        return -1;
    }
    cout << "\n====================================================\n";
    cout << "Setting up the YARP connection...\n";
    RpcClient port; // **** PORT - CLIENT
    port.open("/worldManipulator");
    yarp.connect("/worldManipulator","/icubSim/world");
    
    RpcServer port2; // **** PORT - SERVER
    port2.open("/worldManipulator/targetNum"); // RpcClient port_makeObj ("/controller/targetNum") gets connected to this server and send a number b/w 0-45
    
    
    double objPosRot1[5]; // ON XY plane of the task space, [0]: X Pos., [1]: Y Pos., [2]: Landscape/Portrait., [3]: Obj. Orientation
    double randObj[5]; // ON XY plane of the task space, [0]: X Pos., [1]: Y Pos., [2]: Landscape/Portrait., [3]: Obj. Orientation
    
    while(1)
    {
        if(port.getOutputCount() == 0) // This function returns no. of output ports this port is connected to. In this case it should return 1.
        {
            fprintf(stderr,"/icubSim/world Module is NOT connected... Check the connection\n");
            return 0;
        }
        else
        {
            Bottle cmd_targetNum;
            Bottle response_targetNum;
            
            double desiredTargetNum1; // First Obj.
            double desiredTargetNum2; // First Obj.
            
            port2.read(cmd_targetNum, true); // RPC Server waiting for the input
            desiredTargetNum1 = cmd_targetNum.get(0).asDouble();
            desiredTargetNum2 = cmd_targetNum.get(1).asDouble();
            //cout <<"Received: Target Number 1 = " << desiredTargetNum1 << "\n";
            // Before we make the object, we always delete the entire object in the simulation.
            Bottle cmd;
            Bottle response;
            cmd.addString("world");
            cmd.addString("del");
            cmd.addString("all");
            port.write(cmd,response); //wait until we get the response
            
            if(desiredTargetNum1 != 0 && desiredTargetNum2 != 0)
            {
                cout <<"Received: " << desiredTargetNum1 << "\t" << desiredTargetNum2 <<"\n";
                
                // Making the table
                Bottle cmd2;
                cmd2.addString("world");
                cmd2.addString("mk");
                cmd2.addString("sbox");
                cmd2.addDouble(TABLE_WIDTH);
                cmd2.addDouble(TABLE_THICKNESS);
                cmd2.addDouble(TABLE_DEPTH);
                cmd2.addDouble(TABLE_POS_X);
                cmd2.addDouble(TABLE_POS_Y);
                cmd2.addDouble(TABLE_POS_Z);
                cmd2.addDouble(TABLE_COLOR_R);
                cmd2.addDouble(TABLE_COLOR_G);
                cmd2.addDouble(TABLE_COLOR_B);
                port.write(cmd2,response); //wait until we get the response
                
                
                
                
                
                // Making the first object
                getObjPosRot((int)desiredTargetNum1,objPosRot1); // calculates the Object position and orientation.
                int type1st = (desiredTargetNum1 - ((int)desiredTargetNum1%1000)) / 1000;
                Bottle cmd3;
                cmd3.addString("world"); // world mk box (box dimensions) (3D position) (RGB)
                cmd3.addString("mk");
                
                if(type1st == 1 || type1st == 4) // Ball
                {
                    cmd3.addString("sph");
                    cmd3.addDouble(objPosRot1[2]);
                }
                else if(type1st == 2 || type1st == 5) // Box
                {
                    cmd3.addString("box");
                    cmd3.addDouble(objPosRot1[2]);
                    cmd3.addDouble(objPosRot1[3]);
                    cmd3.addDouble(objPosRot1[4]);
                }
                else if(type1st == 3 || type1st == 6) // Cylinder
                {
                    cmd3.addString("cyl");
                    cmd3.addDouble(objPosRot1[2]);
                    cmd3.addDouble(objPosRot1[3]);
                }
                
                cmd3.addDouble(objPosRot1[0]);
                cmd3.addDouble(OBJ_POS_Y);
                cmd3.addDouble(objPosRot1[1]);
                cmd3.addDouble(OBJ_1_R);
                cmd3.addDouble(OBJ_1_G);
                cmd3.addDouble(OBJ_1_B);
                port.write(cmd3,response); //wait until we get the response
                
                if(type1st == 3 || type1st == 6) // Cylinder
                {
                    cmd3.clear();
                    cmd3.addString("world"); // world rot box (box num) (rotx roty rotz)
                    cmd3.addString("rot");
                    cmd3.addString("cyl");
                    cmd3.addInt(1);
                    cmd3.addDouble(90);
                    cmd3.addDouble(0);
                    cmd3.addDouble(0);
                    port.write(cmd3,response); //wait until we get the response
                    Time::delay(0.1);
                }
                else if(type1st == 2 || type1st == 5) // BOX
                {
                    cmd3.clear();
                    cmd3.addString("world"); // world rot box (box num) (rotx roty rotz)
                    cmd3.addString("rot");
                    cmd3.addString("box");
                    cmd3.addInt(1);
                    cmd3.addDouble(0);
                    //cmd3.addDouble(objPosRot1[4]);
                    cmd3.addDouble(90);
                    cmd3.addDouble(0);
                    port.write(cmd3,response); //wait until we get the response
                    Time::delay(0.1);
                }
                
                
                Time::delay(1); // Wait until the object is dropped on the table.
                
                cmd3.clear();
                response.clear();
                cmd3.addString("world"); // world rot box (box num) (rotx roty rotz)
                cmd3.addString("get");
                
                if(type1st == 1 || type1st == 4) // Ball
                    cmd3.addString("sph");
                else if(type1st == 2 || type1st == 5) // Box
                    cmd3.addString("box");
                else if(type1st == 3 || type1st == 6) // Cylinder
                    cmd3.addString("cyl");
                cmd3.addInt(1);
                port.write(cmd3,response); //wait until we get the response
                
                cout << response.size() << "\t" << response.toString() << "\n";
                double x = response.get(0).asDouble();
                double y = response.get(1).asDouble();
                double z = response.get(2).asDouble();
                
                response_targetNum.addDouble(x);
                response_targetNum.addDouble(y);
                response_targetNum.addDouble(z);
                response_targetNum.addDouble(objPosRot1[4]); // Orientation
                
                Time::delay(0.5);
                
                
                
            }
            else
            {
                response_targetNum.addDouble(0);
            }
            port2.reply(response_targetNum);
            Time::delay(0.1);
        }
    }
    return 0;
    
}


void getObjPosRot(int seqNum, double *objInfo)
{
    int objType = (seqNum - (seqNum%1000))/1000;
    int objSize = ((seqNum%1000) - (seqNum%100))/100;
    int objLoc = ((seqNum%100) - (seqNum%10))/10;
    int objOri = (seqNum%10);
    
    if(objType > 3) // random obj
    {
        char randFileName[256];
        sprintf(randFileName, "./trialConf/randomTrials/randomTrial_%04d.txt",seqNum);
        FILE *trialFP;
        trialFP = fopen(randFileName, "r");
        
        
        fscanf(trialFP, "%lf", &objInfo[0]);   // x
        fscanf(trialFP, "%lf", &objInfo[1]);   // y
        fscanf(trialFP, "%lf", &objInfo[2]);   //
        fscanf(trialFP, "%lf", &objInfo[3]);   //
        fscanf(trialFP, "%lf", &objInfo[4]);   //
        
        fclose(trialFP);
        
    }
    else
    {
        
        switch(objType)
        {
            case 1:
                if(objSize == 1) // Big
                {
                    objInfo[2] = OBJ_1_BIG_RAD;
                }
                else
                {
                    objInfo[2] = OBJ_1_SMALL_RAD;
                }
                break;
            case 2:
                if(objSize == 1) // Big
                {
                    objInfo[2] = OBJ_2_BIG_X;
                    objInfo[3] = OBJ_2_BIG_Y;
                    objInfo[4] = OBJ_2_BIG_Z;
                }
                else
                {
                    objInfo[2] = OBJ_2_SMALL_X;
                    objInfo[3] = OBJ_2_SMALL_Y;
                    objInfo[4] = OBJ_2_SMALL_Z;
                }
                break;
            case 3:
                if(objSize == 1) // Big
                {
                    objInfo[2] = OBJ_3_BIG_RAD;
                    objInfo[3] = OBJ_3_BIG_LENGTH;
                }
                else
                {
                    objInfo[2] = OBJ_3_SMALL_RAD;
                    objInfo[3] = OBJ_3_SMALL_LENGTH;
                }
                break;
        }
        
        switch(objLoc)
        {
            case 1:
                objInfo[0] = POS_L_1_X;
                objInfo[1] = POS_L_1_Y;
                break;
            case 2:
                objInfo[0] = POS_L_2_X;
                objInfo[1] = POS_L_2_Y;
                break;
            case 3:
                objInfo[0] = POS_L_3_X;
                objInfo[1] = POS_L_3_Y;
                break;
            case 4:
                objInfo[0] = POS_L_4_X;
                objInfo[1] = POS_L_4_Y;
                break;
            case 5:
                objInfo[0] = POS_R_1_X;
                objInfo[1] = POS_R_1_Y;
                break;
            case 6:
                objInfo[0] = POS_R_2_X;
                objInfo[1] = POS_R_2_Y;
                break;
            case 7:
                objInfo[0] = POS_R_3_X;
                objInfo[1] = POS_R_3_Y;
                break;
            case 8:
                objInfo[0] = POS_R_4_X;
                objInfo[1] = POS_R_4_Y;
                break;
            case 9:
                objInfo[0] = POS_R_9_X;
                objInfo[1] = POS_R_9_Y;
                break;
            case 0:
                objInfo[0] = POS_L_0_X;
                objInfo[1] = POS_L_0_Y;
                break;
            default:
                objInfo[0] = POS_R_4_X;
                objInfo[1] = POS_R_4_Y;
                break;
        }
        

    }
}
