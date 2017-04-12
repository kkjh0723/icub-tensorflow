#include <stdio.h>
#define TESTING_ONLINE
/////////////////////////////////////////////////////
// Configure Dataset.
//#define NUM_OF_CLASSES 5	//number of classes // not used anymore in the new style
//#define NUM_OF_PERSONS 6 	//number of persons // not used anymore in the new style
//#define NUM_OF_TRIALS 1		//number of trials of spefic action per each person
// Target File Type
#define OUTPUT_DIMENSION 10
#define SOFTMAX_DIMENSION 10 // NUMBER OF SOFTMAX_DIMENSION PER OUTPUT
//#define OUTPUT_NUMBER_FEATUREMAP (OUTPUT_DIMENSION * SOFTMAX_DIMENSION)
#define NUM_OF_OUTPUT (OUTPUT_DIMENSION * SOFTMAX_DIMENSION)
///////////////////////////////////////////////////////

#define NORMALIZED_DATASET
///////////////////////////////////////////////////////
// Regarding the vision input while the network outputs the motor,
#define DYNAMIC_VISION_INPUT_WHILE_MOTOR_OUTPUTS
//#define BLACK_VISION_INPUT_WHILE_MOTOR_OUTPUTS
//#define STATIC_VISION_INPUT_WHILE_MOTOR_OUTPUTS

// applying noise to the vision input while generating the motor outputs
//#define APPLY_SALT_PEPPER_NOISE // Works for both static & dynamic cases
#ifdef APPLY_SALT_PEPPER_NOISE
	#define TH_SALT 0.99 // The higher, the less noise
	#define TH_PEPPER 0.98
#endif
///////////////////////////////////////////////////////


///////////////////////////////////////////////////////
// Regarding the motor teaching signal while receiving vision input
#define MOTOR_TEACHING_WHILE_VISION
#ifdef MOTOR_TEACHING_WHILE_VISION
	#define VISION_LENGTH 0//44
#else
	#define VISION_LENGTH 0
#endif
///////////////////////////////////////////////////////



#define IMG_ROW 48
#define IMG_COL 64

// Type of activation function used in layers except OUTPUT
// Output Layer: Softmax Activation
//#define SIGMOID
//#define SOFTSIGN	// Warning! This might need to be revised.
#define SIGMOID_VAR // from Minju's model
//#define MODIFIED_LOGISTIC_TRANSFER // From Heinrich, Weber and Wermter (2013) "Embodied Language Understanding..."


typedef struct{
	int seqSize;
	int *stepSize;		//[# of seq]
	
	double ***img;		//[# of seq][# of step][# of pixel]
	//int **label;		//[# of seq][# of class]
	double ***label;		//[# of seq][# of class][# of motor seq]
	int classNumber;
}InputData;

typedef struct{
	int row;
	int col;
}MatSize;

int check_directory(char *path);
void recordAccuracy(double accuracy, int **confusionMat, int **resPerImg, int **label, int epoch, int seqSize, int *stepSize);
double gaussianRand(void);
double uniformRand(double begin, double end);
void RandomizeIdx(int *idx, int numOfIndex);
InputData * readData_online(int batchSize, int *numOfBatch, int idxConfigFile, bool isTrain, double *recvImg);
void readSeq(InputData *input, int idxBatch, int idxSeq, int idxClass, int idxPerson, int idxTrial);
//void readSeq_RNN(InputData *input, int idxBatch, int idxSeq, int idxClass, int idxPerson, int idxTrial, int idxSeqToRead);
void readSeq_RNN_online(InputData *input, int idxBatch, int idxSeq, int idxSeqToRead ,double *recvImg);
void freeInputData(InputData input);


double activation(double x);
double dActivation(double x);
