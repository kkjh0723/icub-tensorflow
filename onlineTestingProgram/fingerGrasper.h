/// For Converting GRASPING LEVEL TO EACH FINGER'S REAL JOINT VALUES
#define GRASPING_LEVEL_MIN 1
#define GRASPING_LEVEL_MAX 10

#define FINGER_0_MIN	40//60//15
#define FINGER_0_MAX	40//60//22

#define FINGER_1_MIN	40//90//40
#define FINGER_1_MAX	90

#define FINGER_2_MIN	0
#define FINGER_2_MAX	1

#define FINGER_3_MIN	0
#define FINGER_3_MAX	70// 27

//Index
#define FINGER_4_MIN	20
#define FINGER_4_MAX	70///60//40 //60

#define FINGER_5_MIN	10//20
#define FINGER_5_MAX	80//70///60//40// 51

// MID
#define FINGER_6_MIN	20
#define FINGER_6_MAX	70///60
#define FINGER_7_MIN	10
#define FINGER_7_MAX	80///60//40//49

// 4 & 5th
#define FINGER_8_MIN	60
#define FINGER_8_MAX	140//180//140


#define MINIMUM_GRASPING_SENSOR_VALUE 130 // 255 is the maximum (all fingers are touching the object firmly)
#define MAX_NUM_TRY 50 // if it tries more than this val, it stops.

/// ROBOT CONFIGURATION
#define JNT_SPD 10//39 // 10 IS ENOUGH FOR TESTING... 
#define JNT_ACC 10//24 // 5 IS ENOUGH FOR TESTING...
#define NUM_JOINT_ARM 7
#define NUM_JOINT_FINGERS 9
