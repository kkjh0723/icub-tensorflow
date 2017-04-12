// For Network Variables. go utils.h
#define NUM_TRIALS 2
//#define SAVE_WORLD_VIEW

//#define CAPTURE_SIM
#define IDX_GRASPING_LEVEL 9
#define IDX_ZOOM_LEVEL 10

#define ZOOM
//#define USE_DATASET_VISION

/// ROBOT CONFIGURATION
#define JNT_SPD 200//39 // 10 IS ENOUGH FOR TESTING... 
#define JNT_ACC 10//24 // 5 IS ENOUGH FOR TESTING...

/// VISION CONFIGURATION
#define IMG_ROW 48
#define IMG_COL 64
#define IMG_SIZE (IMG_ROW*IMG_COL)
#define IMG_ROW_WORLD 240
#define IMG_COL_WORLD 320



#define MOTOR_LENGTH 130//140//310
#define NEW_SESSION_BEGINS 154

//#define NUM_JOINT 16 // head(

#define NUM_JOINT_ARM 7
#define NUM_JOINT_FINGERS 9

#define READY 1
#define NOT_READY 0

#define CAPTURE 1
#define NOT_CAPTURE 0

#define CALC 1
#define NOT_CALC 0
#define INIT_NET 2



/// MISC.
//#define DELAY_VISION 0.01
//#define DELAY_NETWORK 0.01
//#define DELAY_CONTROLLER 0.01
#define DELAY_VISION 0.001 //0201
#define DELAY_NETWORK 0.001 //0201
#define DELAY_CONTROLLER 0.001 // 0201

#define STEP_INTERVAL 1


#define RIGHT_ARM 0
#define LEFT_ARM 1
#define TORSO 2
#define HEAD 3

// Head(2: JNT 0 & JNT 2)
// ARM(7: Jnt 0 ~ Jnt 6)
// GRASPING_LEVEL (1: 0=straight fingers, 1 = grasp)
// SUM = 10;
#define NUM_SIG_DIM 10
#define SOFTMAX_DIMENSION 10
#define TOTAL_SOFTMAX_DIM (NUM_SIG_DIM*SOFTMAX_DIMENSION)



#define TARGET_1ST	0
#define OTHER_1ST	1
#define GESTURE_1ST	2
#define TARGET_2ND	3
#define OTHER_2ND	4
#define GESTURE_2ND	5
#define CLASSNUM 	6


/// FOR SOFTMAX TRANSFORM (SOFTMAX OUTPUT -> ANALOG VALUE)
#define X_LENGTH 1001
#define INVX_LENGTH 11


#define OBS_X_MIN	-0.32
#define OBS_X_MAX	-0.22
#define ROT_MIN -45
#define ROT_MAX 45
#define OBJ_X_MIN -0.04
#define OBJ_X_MAX -0.04
#define OBJ_Y_MIN 0.2
#define OBJ_Y_MAX 0.3

