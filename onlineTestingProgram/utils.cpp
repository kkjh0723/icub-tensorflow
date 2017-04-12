#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "utils.h"

using namespace std;

int check_directory(char *path){
	struct stat st = {0};

	if (stat(path, &st) == -1) {
		if(mkdir(path, 0700) == 0){
			//succeed to make dir
			return 1;
		}else{
			//fail to make dir
			return -1;
		}
	}else{
		//directory already exist
		return 0;
	}
}

void recordAccuracy(double accuracy, int **confusionMat, int **resPerImg, int **label, int epoch, int seqSize, int *stepSize){
	// Not Used
}

double gaussianRand(void) {
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if (phase == 0) {

    do {
        double U1 = (double)rand() / RAND_MAX;
        double U2 = (double)rand() / RAND_MAX;
        V1 = 2 * U1 - 1;
        V2 = 2 * U2 - 1;
        S = V1 * V1 + V2 * V2;
    } while (S >= 1 || S == 0);

    X = V1 * sqrt(-2 * log(S) / S);

    } else X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;
    return X;
}

// Uniform random generator (provided in c++)
double uniformRand(double begin, double end) 
{
  double number = rand() / (RAND_MAX + 1.0) * (end - begin) + begin;
  return number;
}








InputData * readData_online(int batchSize, int *numOfBatch, int idxConfigFile, bool isTrain, double *recvImg)
{
	int idxTestPerson = idxConfigFile;
	
	// Read the training set configuration file.
	
	int numberOfSeq = 1;
	(*numOfBatch) = numberOfSeq;
	InputData *Data = new InputData[(*numOfBatch)];
	int dataSize = numberOfSeq;
	int *idxData = new int[dataSize];
	for(int idx = 0; idx < dataSize; idx++)	idxData[idx] = idx;
	int idxSeq = 0;
	int idxBatch = 0;
	for(idxBatch = 0; idxBatch < (*numOfBatch); idxBatch++)
	{
		Data[idxBatch].seqSize = batchSize;
		Data[idxBatch].stepSize = new int[batchSize];
		Data[idxBatch].img = new double**[batchSize];
	}
	idxBatch = 0;
	
	for(int idx = 0; idx < dataSize; idx++)
	{
		int idxSeqToRead = 0;
		readSeq_RNN_online(Data, idxBatch, idxSeq, idxSeqToRead, recvImg);		//read single sequence in the DataSet
		idxSeq++;
		if(idxSeq == batchSize){	//after read one batch
			idxSeq = 0;
			idxBatch++;
		}
	}
	//free(idxData);
	delete[] idxData;
	return Data;
}


void readSeq_RNN_online(InputData *input, int idxBatch, int idxSeq, int idxSeqToRead ,double *recvImg)
{
	int tmpIdxSeqToRead = idxSeqToRead;
	int frameSize;
	int realFrameSize;
	frameSize = 1; 
	input[idxBatch].stepSize[idxSeq] = frameSize;
	input[idxBatch].img[idxSeq] = new double*[frameSize];
	input[idxBatch].classNumber = idxSeqToRead;

	// Read images (from the txt files)
	for (int idxStep = 0; idxStep < frameSize; idxStep++)
	{
		input[idxBatch].img[idxSeq][idxStep] = new double[IMG_ROW * IMG_COL];

		int idxPixel = 0;
		if(idxStep < frameSize)
		{
			while(idxPixel < (IMG_ROW * IMG_COL))
			{
				//fscanf (fp , "%lf", &input[idxBatch].img[idxSeq][idxStep][idxPixel]);
				input[idxBatch].img[idxSeq][idxStep][idxPixel] = recvImg[idxPixel];
				//cout <<input[idxBatch].img[idxSeq][idxStep][idxPixel]<<"\n";
				
				//getchar();
			#ifdef NORMALIZED_DATASET
				input[idxBatch].img[idxSeq][idxStep][idxPixel] = (input[idxBatch].img[idxSeq][idxStep][idxPixel] - 0.5)*2;
			#endif	
				idxPixel++;
			}
		}
	}
}



void readSeq(InputData *input, int idxBatch, int idxSeq, int idxClass, int idxPerson, int idxTrial){
////////////////////////////////////////
// This function is deprecated.
////////////////////////////////////////
}



void freeInputData(InputData input){

	for(int seq = 0; seq < input.seqSize; seq++){
		for(int step = 0; step < input.stepSize[seq]; step++)
		{
			delete[] input.img[seq][step];
		}

		delete[] input.img[seq];
#ifndef TESTING_ONLINE	
		delete[] input.label[seq];
#endif		
	}
	
	delete[] input.img;
#ifndef TESTING_ONLINE	
	delete[] input.label;
#endif		
	delete[] input.stepSize;
} 

void RandomizeIdx(int *idx, int numOfIndex)
{
	int i, j;

	// DO NOT Set the seed here..
	//srand((unsigned)time(0));
	//srand(1234);

	for(i=0; i<numOfIndex; i++)
	{
		j = int((double)rand()/(double)RAND_MAX*double(numOfIndex-1));

		int temp = idx[i];
		idx[i] = idx[j];
		idx[j] = temp;
	}
}


#ifdef MODIFIED_LOGISTIC_TRANSFER
double activation(double x){
	return (1.7159 / (1+exp(-0.92 * x))) - 0.35795;
}

double dActivation(double x){
	double temp = exp(-0.92 * x);
	return (1.578628 * temp) / ((1+temp) * (1+temp));
}
#endif

/*
#ifdef SIGMOID_VAR
inline double activation(double x){
	return 1.7159 * tanh(0.66666667 * x);
}

inline double dActivation(double x){
	return 0.66666667/1.7159*(1.7159+(x))*(1.7159-(x));
}
#endif
*/

#ifdef SOFTSIGN
double activation(double x){
	return x / ( 1 + fabs(x) );
}

// WARNING. Please confirm it using Desmos graphing calculator
double dActivation(double x){
	return 1 / ((1 + abs(x)) * (1 + abs(x)));
}
#endif

#ifdef SIGMOID
//sigmoid function
double activation(double x)
{
    double y;
    y=(double)1/(1+exp(-(x)*1.0));
    return y;
}
// derivative of sigmoid function
double dActivation(double x)
{
    return x * (1.0 - x);
}
#endif

