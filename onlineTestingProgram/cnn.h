#pragma once
#include <math.h>
#include "utils.h"
typedef unsigned char uchar;
////////////////////////////////////////////////////////
//#define RUN_ON_SERVER
#ifndef RUN_ON_SERVER
	#include <opencv2/opencv.hpp>
	#define USE_OPENCV // if defined, openCV is used to plot.
#endif
////////////////////////////////////////////////////////
// For Testing the network, please comment out the following line.
//#define TRAINING // Always comment out when testing online.
#ifndef TRAINING
	#define TESTING_ONLINE
#endif
#define ALLOW_PLOT // To plot (openCV)
#ifdef TRAINING
#else
	#define PRINT_RESULT_INTERVAL 1 // every N epoch, the result is printed out.
	#define PRINT_CNN_FAST
	#define PRINT_CNN_MID
	#define PRINT_CNN_SLOW
	#define PRINT_RNN_CTX
	#define PRINT_RNN_CTX_FAST
	//#define PRINT_RNN_OUTPUT
#endif
///////////////////////////////////////////////////////
// Weight Initialization Methods
//#define INITIALIZE_METHOD_1 // Ref:Understanding the difﬁculty of training deep feedforward neural networks by Glorot & Bengio
//#define INITIALIZE_METHOD_2 // Ref:Emergence of Functional Hierarchy in a Multiple Timescale Neural Network Model: A Humanoid Robot Experiment
#define INITIALIZE_METHOD_3 //Using Gaussian Random.

////////////////////////////////////////////////////////
// Configure CNN & RNN Parameters (number of neurons) here. Tau is set in cnn.cpp
#define SMALL_KERNEL
#define NUM_OF_MSTNN_FAST 4
#define NUM_OF_MSTNN_SLOW 8
#define NUM_OF_PFC 20
#define NUM_OF_MTRNN_SLOW 30	// Number of neurons in the RNN's middle context 
#define NUM_OF_MTRNN_FAST 50	// Number of neurons in the RNN's fast context 


//#define CLAMPING // If defined, several neurons in RNN do not receive input from CNN Slow
#ifdef CLAMPING
	#define NUM_NEURON_CONNECTED 30
	#define NUM_NEURON_NOT_CONNECTED (NUM_OF_RNN_CTX_MID - NUM_NEURON_CONNECTED) // First N featuremap would not have connection to CNN Slow
#endif
////////////////////////////////////////////////////////

////////////////////////////////////////////////////////
// Misc. Parameters
////////////////////////////////////////////////////////
#define USE_SOFTMAX_OUTPUT
#define SEED_NUMBER 1234

//////////////////////////////////////////////////////////////////////////////////////////////
// Note that, when printing out the activation value of each context, layer index was used.
// i.e. if the layer is added/deleted/modified, check that part.
// Layers
#define INPUT_LAYER		0	
#define CONVOLUTIONAL	1	//calculate next layer by convoluting
#define FULLY_CONNECTED 2	//current & next layer are fully-connected
#define OUTPUT_LAYER 	3	// Used in CNN

#define RNN_OUTPUT 		6

#define MTRNN_FAST 	4
#define RNN_CONTEXT 	MTRNN_FAST
#define MTRNN_SLOW 	5
#define RNN_CTX_BND 	MTRNN_SLOW

#define PFC 		7




// SOURCE (Used at FM Calculate/Backpropagate for RNN)
#define PREVIOUS 1
#define CURRENT 2
#define NEXT 3
//////////////////////////////////////////////////////////////////////////////////////////////


class Layer;

class FeatureMap
{
public:
	
	double **kernel, bias; // kernel represents weight from the previous later to currrent
	double **kernel_next, **kernel_self; // for RNN, kernel_next refers to the weight from the next layer to the current layer
	// Kernel[SRC][DEST]
	
	double ***inter, ***value;
	double ***cnnSlowValue; // This is to hold the CNN Slow Activation Value over steps.
	double ***cnnSlowValue_testing; // This is to hold the CNN Slow Activation Value over steps.
	// Value[]
	
	double **dErr_wrtb, ***dError, ****dErr_wrtw;
	double ****dErr_wrtw_next,****dErr_wrtw_self;
	// dErr[seq][step][preMap][FMSIZE]
	
	int m_nFeatureMapPrev;
	int m_nFeatureMapNext; // for RNN
	Layer *pLayer;

	void Construct();
	void Construct_RNN();
	void Delete();
	void AllocWRTCal(int seqSize, int *stepSize, int numOfBatch);
	void DeleteWRTCal(int seqSize, int *stepSize);
	
	#ifdef USE_OPENCV
	cv::Mat Plot(int seq, int step, MatSize mapSize, int type);
	#endif
	
	void Clear(const int seqSize, const int *stepSize);	
	double Convolute(double *input, const MatSize &size, const unsigned int &r0, const unsigned int &c0, double *weight, const MatSize &kernel_size);
	void Calculate(double *valueFeatureMapPrev, const int &idxFeatureMapPrev, const int &seq, const int &step, const int &SOURCE);
};

class Layer
{
public:
	int m_type;
	int m_nFeatureMap;
	MatSize m_FeatureSize;
	MatSize m_KernelSize;
	int m_SamplingFactor;
	double m_eta;			// 1/timeconst
	
	int m_layerIndex;
	Layer *pLayerPrev;
	Layer *pLayerNext;	// This is for the context layers in RNN
	FeatureMap* m_FeatureMap;

	void Calculate(const int seq, const int step, const int epoch, const int batchIdx, const int classNumber);
	#ifdef USE_OPENCV
	cv::Mat Plot(int seq, int step);
	#endif
	void Construct(int type, int nFeatureMap, int FeatureSizeRow, int FeatureSizeCol, int KernelSizeRow, int KernelSizeCol, int SamplingFactor, double timeConst);
	void Construct_RNN(int type, int nFeatureMap, int FeatureSizeRow, int FeatureSizeCol, int KernelSizeRow, int KernelSizeCol, int SamplingFactor, double timeConst);
	void Delete();
};

class CCNN
{
public:
	CCNN(void);
	~CCNN(void);

	Layer *m_Layer;
	int m_nLayer;

	void ConstructNN();
	void DeleteNN();
	void LoadWeights(char const *FileName);
	void AllocWRTCal(int seqSize, int *stepSize, int numOfBatch);
	void DeleteWRTCal(int seqSize, int *stepSize);
	void Calculate_online(double *recvImg, double *netOut, int globalStep, double xPos, double yPos, double angle, double xObs); // with Online
	void CalculateStep_online(double *input, int seq, int step, int globalStep);
	void Plot(int seq, int step, double xPos, double yPos, double angle, double xObs, int globalStep);

	int m_classNumber;
	void setClassNumber(int classNumber);
	int getClassNumber(void);	
};


