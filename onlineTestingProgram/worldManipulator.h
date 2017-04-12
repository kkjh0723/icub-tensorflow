/* Revised for VRPGP2016


*/

// dimensions of the static table: world mk sbox 1 0.01 0.5 0 0.48 0.4 1 1 1

/// TABLE
#define TABLE_WIDTH 1
#define TABLE_THICKNESS 0.01
#define TABLE_DEPTH 0.5
#define TABLE_POS_X 0
#define TABLE_POS_Y 0.50//0.48
#define TABLE_POS_Z 0.4
#define TABLE_COLOR_R 1
#define TABLE_COLOR_G 1
#define TABLE_COLOR_B 0

// Dimensions of the red colored box(rgb: 100)


#define OBJ_POS_Y 0.8 // This is height in Sim.


/// Obj: Ball : 1XXX
#define OBJ_1_SMALL_RAD	0.025 // radius
#define OBJ_1_BIG_RAD	OBJ_1_SMALL_RAD*1.4 // radius

#define OBJ_1_R	1
#define OBJ_1_G	0
#define OBJ_1_B	0

/// Obj: Box (Long) : 2XXX
#define OBJ_2_SMALL_X	0.028
#define OBJ_2_SMALL_Y	0.05
#define OBJ_2_SMALL_Z	0.06//0.08//used in the recording

#define OBJ_2_BIG_X	OBJ_2_SMALL_X*1.4 // depth
#define OBJ_2_BIG_Y	OBJ_2_SMALL_Y*1//.4 //height
#define OBJ_2_BIG_Z	OBJ_2_SMALL_Z*1.2 // length

#define OBJ_2_R	1
#define OBJ_2_G	0
#define OBJ_2_B	0


/// Obj: Cylinder : 3XXX
#define OBJ_3_SMALL_RAD	0.025 // radius
#define OBJ_3_SMALL_LENGTH	0.07 // length
#define OBJ_3_BIG_RAD	OBJ_3_SMALL_RAD*1.3// radius
#define OBJ_3_BIG_LENGTH	OBJ_3_SMALL_LENGTH*1.1 // length

#define OBJ_3_R	1
#define OBJ_3_G	0
#define OBJ_3_B	0







/// X & Y coordinates of the various positions on the table.
// Shifted a bit cause we're using the left arm this time.
#define POS_1_X	0.14//+0.1
#define POS_2_X	0.08//+0.1
#define POS_3_X	0.02//+0.1
#define POS_4_X	-0.04//+0.1

#define POS_5_X	0.04
#define POS_6_X	-0.02
#define POS_7_X	-0.08

#define POS_1_Y	0.420 // NOT USED
#define POS_2_Y	0.360 // NOT USED
#define POS_3_Y	0.300
#define POS_4_Y	0.240

#define POS_5_Y	0.270

// Additional
#define POS_0_X	0.11
#define POS_9_X	-0.01

#define POS_9_Y	0.270
#define POS_0_Y	0.270


#define POS_L_0_X POS_0_X
#define POS_L_0_Y POS_0_Y
#define POS_R_9_X POS_9_X
#define POS_R_9_Y POS_9_Y


#define POS_L_1_X POS_1_X
#define POS_L_2_X POS_2_X
#define POS_L_3_X POS_1_X
#define POS_L_4_X POS_2_X
#define POS_R_1_X POS_3_X
#define POS_R_2_X POS_4_X
#define POS_R_3_X POS_3_X
#define POS_R_4_X POS_4_X

#define POS_L_1_Y POS_3_Y
#define POS_L_2_Y POS_3_Y
#define POS_L_3_Y POS_4_Y
#define POS_L_4_Y POS_4_Y

#define POS_R_1_Y POS_3_Y
#define POS_R_2_Y POS_3_Y
#define POS_R_3_Y POS_4_Y
#define POS_R_4_Y POS_4_Y


// 5 Orientations. These are rot in Y-direction
/*
#define ROT_1	90 		//45
#define ROT_2	67.5 	//22.5
#define ROT_3	45		//0
#define ROT_4	22.5	//-22.5
#define ROT_5	0		//-45
#define ROT_6	-22.5	//-22.5
#define ROT_7	-45		//0
#define ROT_8	-67.5 	//22.5
*/
#define ROT_1	90 		//45
#define ROT_2	90//45
#define ROT_3	90//0
#define ROT_4	90//22.5	//-22.5
#define ROT_5	90//0		//-45
#define ROT_6	90//-22.5	//-22.5
#define ROT_7	90//-45		//0
#define ROT_8	90//-67.5 	//22.5

