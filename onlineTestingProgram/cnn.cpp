#include <stdlib.h>
#include <string.h>
#include "cnn.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;


inline double activation(double x){
	return 1.7159 * tanh(0.66666667 * x);
}

inline double dActivation(double x){
	return 0.66666667/1.7159*(1.7159+(x))*(1.7159-(x));
}

CCNN::CCNN(void)
{
	ConstructNN();
}

CCNN::~CCNN(void)
{
	DeleteNN();
}


void CCNN::setClassNumber(int classNumber)
{
	m_classNumber = classNumber;
}

int CCNN::getClassNumber(void)
{
	return m_classNumber;
}


////////////////////////////////////
void CCNN::ConstructNN()
////////////////////////////////////
{
	int i, j, k;
	
	m_nLayer = 7;
	m_Layer = new Layer[m_nLayer];

	m_Layer[0].pLayerPrev = NULL;
	m_Layer[0].pLayerNext = NULL;
	
	for(i=1; i<m_nLayer; i++){ 
		m_Layer[i].pLayerPrev = &m_Layer[i-1];
		m_Layer[i].pLayerNext = NULL;
	}
	
	//Consturct CNN
							//		type		| # of Maps 			| Map size 				| Kernel size | Sub-sampling | Time constant
	m_Layer[0].Construct	(	INPUT_LAYER,		1,						IMG_ROW,IMG_COL,		0,0,			0,			1);
	
  m_Layer[1].Construct	(	CONVOLUTIONAL,		NUM_OF_MSTNN_FAST,		21,29,					8,8,			2,			1);
	m_Layer[2].Construct	(	CONVOLUTIONAL,		NUM_OF_MSTNN_SLOW,		8,12,					7,7,			2,			5);
	m_Layer[3].Construct	(	PFC,				      NUM_OF_PFC,				1,1,					8,12,			1,			25);
	
	m_Layer[4].Construct	(	MTRNN_SLOW,			NUM_OF_MTRNN_SLOW,		1,1,					1,1,			1,			10); 
	m_Layer[5].Construct	(	MTRNN_FAST,			NUM_OF_MTRNN_FAST,		1,1,					1,1,			1,			2); 
	m_Layer[6].Construct	(	RNN_OUTPUT,			NUM_OF_OUTPUT,			1,1,					1,1,			1,			1); 

	// Contruct_RNN: setting the link from the next layer &  also self link
	m_Layer[3].pLayerNext = &m_Layer[4]; // PFC - MTRNN_SLOW
	m_Layer[3].Construct_RNN	(	PFC,		NUM_OF_PFC,			1,1,	2,4,	1,	16); // PFC
	m_Layer[4].pLayerNext = &m_Layer[5]; // MTRNN_SLOW - MTRNN_FAST
	m_Layer[4].Construct_RNN	(	MTRNN_SLOW,	NUM_OF_MTRNN_SLOW,	1,1,	1,1,	1,	4); // RNN_MOTOR: Middle
	m_Layer[5].Construct_RNN	(	MTRNN_FAST,	NUM_OF_MTRNN_FAST,	1,1,	1,1,	1,	1); // RNN_MOTOR: Output

	for(i = 0 ; i < m_nLayer ; i++){
		m_Layer[i].m_layerIndex = i;
	}



	ofstream SaveFile2("./result/log_configuration.txt", ios::app);
	SaveFile2.precision(0);
	SaveFile2<<std::fixed;
	SaveFile2 << "Layer Types:\n";
	SaveFile2 << "INPUT_LAYER\t0\n"
		<< "CONVOLUTIONAL\t1 \n"
		<< "FULLY_CONNECTED\t2 \n"
		<< "MTRNN_FAST\t4 \n"
		<< "MTRNN_SLOW\t5\n"
		<< "RNN_OUTPUT\t6 \n"
		<< "PFC\t7 \n\n";
	
	SaveFile2 << "Number of Layers: " << m_nLayer << "\n";
	for (int idxNlayer = 0 ; idxNlayer < m_nLayer ; idxNlayer++)
	{
		SaveFile2 << "Layer[" << idxNlayer << "]\t"
			<< "Type: " << m_Layer[idxNlayer].m_type << "\t"
			<< "NumberOfFM: " << m_Layer[idxNlayer].m_nFeatureMap << "\t"
			<< "FMSize: " << m_Layer[idxNlayer].m_FeatureSize.row << "x" << m_Layer[idxNlayer].m_FeatureSize.col << "\t"
			<< "KernelSize: " << m_Layer[idxNlayer].m_KernelSize.row << "x" << m_Layer[idxNlayer].m_KernelSize.col << "\t"
			<< "SamplingFactor: " << m_Layer[idxNlayer].m_SamplingFactor << "\t"
			<< "TimeConstant: " << 1 / m_Layer[idxNlayer].m_eta << "\n";
	}


SaveFile2 << "\n============================================================\n";
#ifdef CLAMPING
		SaveFile2 << "CLAMPING: Among " << NUM_OF_MTRNN_SLOW << " RNN neurons, " << NUM_OF_MTRNN_SLOW - NUM_NEURON_NOT_CONNECTED << " neurons are receiving inputs from CNN Slow.\n";
#else
		SaveFile2 << "CLAMPING: Fully connected between CNN SLOW and RNN \n";
#endif
SaveFile2.precision(6);
SaveFile2 << "SEED: " << SEED_NUMBER << "\n";


#ifdef SIGMOID 
	SaveFile2 << "Activation Function: SIGMOID\n";
#endif
#ifdef SOFTSIGN
	SaveFile2 << "Activation Function: SOFTSIGN\n";
#endif
#ifdef SIGMOID_VAR
	SaveFile2 << "Activation Function: SIGMOID_VAR (MINJU'S VERSION)\n";
#endif
#ifdef MODIFIED_LOGISTIC_TRANSFER
	SaveFile2 << "Activation Function: Modified Logistic Transfer Function (from Heinrich, Weber and Wermter (2013)\n";
#endif


#ifdef USE_SOFTMAX_OUTPUT
	SaveFile2 << "SOFTMAX function at OUTPUT\n";
#else
	SaveFile2 << "Above activation function at OUTPUT\n";
#endif


	SaveFile2 << "============================================================\n";
#ifdef MOTOR_TEACHING_WHILE_VISION	
	SaveFile2 <<  "MOTOR TEACHING WHILE VISION INPUT\n";
#endif	
	SaveFile2 <<  "ACTUAL VISION INPUT LENGTH: " << VISION_LENGTH << "\n";

	SaveFile2 << "============================================================\n";
// Regarding the vision input while the network outputs the motor signals
	SaveFile2 <<  "VISION INPUT WHILE MOTOR OUTPUTS: ";
#ifdef BLACK_VISION_INPUT_WHILE_MOTOR_OUTPUTS
	SaveFile2 <<  "BLACK (0)_VISION_INPUT_WHILE_MOTOR_OUTPUTS \n";
#endif	
#ifdef STATIC_VISION_INPUT_WHILE_MOTOR_OUTPUTS
	SaveFile2 <<  "STATIC_VISION_INPUT_WHILE_MOTOR_OUTPUTS \n";
#endif	
#ifdef DYNAMIC_VISION_INPUT_WHILE_MOTOR_OUTPUTS
	SaveFile2 <<  "DYNAMIC_VISION_INPUT_WHILE_MOTOR_OUTPUTS \n";
#endif
#ifdef APPLY_SALT_PEPPER_NOISE
	SaveFile2 <<  "APPLY_SALT_PEPPER_NOISE to the static vision input \n";
#endif	

#ifdef NORMALIZED_DATASET
	SaveFile2 <<  "Normalized dataset (-0.5 to 0.5)\n";
#endif	

#ifdef INITIALIZE_METHOD_1
	SaveFile2 <<  "INITIALIZE_METHOD_1 // Ref:Understanding the difﬁculty of training deep feedforward neural networks by Glorot & Bengio. \n";
#endif
#ifdef INITIALIZE_METHOD_2
	SaveFile2 <<  "INITIALIZE_METHOD_2 // Ref:Emergence of Functional Hierarchy in a Multiple Timescale Neural Network Model: A Humanoid Robot Experiment. \n";
#endif
#ifdef INITIALIZE_METHOD_3
	SaveFile2 <<  "INITIALIZE_METHOD_3 //Using Gaussian Random.  (Minju's way)\n";
#endif
#ifdef SMALL_KERNEL
	SaveFile2 <<  "SMALL KERNEL \n";
#endif

	SaveFile2 << "============================================================\n";
	SaveFile2.close();
}

///////////////////////////////
void CCNN::DeleteNN()
///////////////////////////////
{
	for(int i=0; i<m_nLayer; i++) m_Layer[i].Delete();
}

//////////////////////////////////////////////
void CCNN::LoadWeights(char const *FileName)
/////////////////////////////////////////////
{
	int i, j, k, m, n;

	FILE *f;
	if((f = fopen(FileName, "r")) == NULL)
	{
		cout << "\nWeight File cannot be loaded. Please make sure that result directory contains 'netInform' file.\n"; getchar();
		exit(1);
		return;
	}

	for ( i=1; i<m_nLayer; i++ )
	{
		//cout <<"LAYER" << i; getchar();
		for( j=0; j<m_Layer[i].m_nFeatureMap; j++ )
		{
			fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].bias);

			for(k=0; k<m_Layer[i].pLayerPrev->m_nFeatureMap; k++)
				for(m=0; m < m_Layer[i].m_KernelSize.row * m_Layer[i].m_KernelSize.col; m++)	
					//fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].kernel_prev[k][m]);
					fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].kernel[k][m]);
	
			if(m_Layer[i].m_type == PFC)
			{
				m = 0;
				for(k=0; k<m_Layer[i].m_nFeatureMap; k++)
					fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].kernel_self[k][m]);
				for(k=0; k<m_Layer[i].pLayerNext->m_nFeatureMap; k++)
					fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].kernel_next[k][m]);
			}
			

			if(m_Layer[i].m_type == MTRNN_SLOW)
			{
				for(k=0; k<m_Layer[i].m_nFeatureMap; k++)
					for(m=0; m < m_Layer[i].m_KernelSize.row * m_Layer[i].m_KernelSize.col; m++)	
					{
						fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].kernel_self[k][m]);
					}
				for(k=0; k<m_Layer[i].pLayerNext->m_nFeatureMap; k++)
					for(m=0; m < m_Layer[i].m_KernelSize.row * m_Layer[i].m_KernelSize.col; m++)	
					{
						fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].kernel_next[k][m]);
					}
			}
			if(m_Layer[i].m_type == MTRNN_FAST)
			{
				for(k=0; k<m_Layer[i].m_nFeatureMap; k++)
					for(m=0; m < m_Layer[i].m_KernelSize.row * m_Layer[i].m_KernelSize.col; m++)	
					{
						fscanf(f, "%lg ", &m_Layer[i].m_FeatureMap[j].kernel_self[k][m]);
					}
			}
	
	
		}
	}
	fclose(f);
}


//////////////////////////////////////////////////////////////////////////
// Calculate the forward dynamics.
void CCNN::Calculate_online(double *recvImg, double *netOut, int globalStep, double xPos, double yPos, double angle, double xObs) // with Online
//////////////////////////////////////////////////////////////////////////
{
	unsigned int idxLayer, idxFM, seq, step;
	int seqSize = 1;
	int stepSize[seqSize];
	stepSize[0] = 2;
	int calcStepBegin = 0;
	int calcStepEnd = 0;
	
	int gStepForPlot = globalStep;
	if(globalStep == 0)
	{
		// Initialize: Clear membrane potential of each feature map of each layer
		for(idxLayer=0; idxLayer<m_nLayer; idxLayer++)
		{
			for(idxFM=0; idxFM< m_Layer[idxLayer].m_nFeatureMap; idxFM++)
			{
				m_Layer[idxLayer].m_FeatureMap[idxFM].Clear(seqSize, stepSize);
			}
		}
		calcStepBegin = 0;
		calcStepEnd = 1;
	}
	else
	{
		calcStepBegin = 1;
		calcStepEnd = 2;
	}
	
	for(step = calcStepBegin; step < calcStepEnd; step++)
	{
		CalculateStep_online(recvImg, 0, step,globalStep); 
		#ifdef ALLOW_PLOT
				if(globalStep == 0)
					Plot(0, 0,xPos,yPos,angle,xObs,globalStep);
				else
					Plot(0, 1,xPos,yPos,angle,xObs,globalStep);
		#endif
	}


	if(globalStep == 0)
	{
		for(idxFM=0; idxFM< m_Layer[m_nLayer-1].m_nFeatureMap; idxFM++)
		{
			netOut[idxFM] = m_Layer[m_nLayer-1].m_FeatureMap[idxFM].value[0][0][0];
		}
	}
	else
	{
		for(idxFM=0; idxFM< m_Layer[m_nLayer-1].m_nFeatureMap; idxFM++)
		{
			netOut[idxFM] = m_Layer[m_nLayer-1].m_FeatureMap[idxFM].value[0][1][0];
		}
	}


#ifdef PRINT_RNN_OUTPUT
		int epoch = 1;
		if(epoch%PRINT_RESULT_INTERVAL == 0)
		{
			string a = "./result/ma"; int b = 0; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str();
			ofstream SaveFile(ans2, ios::app); SaveFile.precision(6); SaveFile<<std::fixed;					
			SaveFile << step << "\t";
			for(idxFM=0; idxFM< m_Layer[m_nLayer-1].m_nFeatureMap; idxFM++)
			{
					SaveFile << netOut[idxFM] <<"\t";
			}
			SaveFile << "\n";
			SaveFile.close();
		}
#endif

	if(globalStep != 0)
	{
		for(int idxLayer=1; idxLayer<6; idxLayer++)
		{
			for(int idxFM=0; idxFM<m_Layer[idxLayer].m_nFeatureMap; idxFM++)
			{
				for(int i = 0; i < m_Layer[idxLayer].m_FeatureSize.row * m_Layer[idxLayer].m_FeatureSize.col; i++)
				{
					m_Layer[idxLayer].m_FeatureMap[idxFM].value[0][0][i] = m_Layer[idxLayer].m_FeatureMap[idxFM].value[0][1][i];
					m_Layer[idxLayer].m_FeatureMap[idxFM].inter[0][0][i] = m_Layer[idxLayer].m_FeatureMap[idxFM].inter[0][1][i];
					m_Layer[idxLayer].m_FeatureMap[idxFM].value[0][1][i] = 0;
					m_Layer[idxLayer].m_FeatureMap[idxFM].inter[0][1][i] = 0;
				}		
			}
		}
		for(int idxLayer=6; idxLayer<7; idxLayer++)
		{
			for(int idxFM=0; idxFM<m_Layer[idxLayer].m_nFeatureMap; idxFM++)
			{
				for(int i = 0; i < m_Layer[idxLayer].m_FeatureSize.row * m_Layer[idxLayer].m_FeatureSize.col; i++)
				{
					m_Layer[idxLayer].m_FeatureMap[idxFM].value[0][0][i] = 0;
					m_Layer[idxLayer].m_FeatureMap[idxFM].inter[0][0][i] = 0;
					m_Layer[idxLayer].m_FeatureMap[idxFM].value[0][1][i] = 0;
					m_Layer[idxLayer].m_FeatureMap[idxFM].inter[0][1][i] = 0;
				}		
			}
		}
	}

}


//////////////////////////////////////////////////////////////////////////
// Step-wise (1step) Forward Dynamics 
void CCNN::CalculateStep_online(double *input, int seq, int step, int globalStep)
//////////////////////////////////////////////////////////////////////////
{
	int i;
	bool calcCNN = true;
	int idxBatch = 0;
	
	int classNumber = getClassNumber();//0;
	int epoch = 1;
	if(globalStep == 0)
	{
		for(i = 0; i < m_Layer[0].m_FeatureSize.row * m_Layer[0].m_FeatureSize.col; i++)
		{
			m_Layer[0].m_FeatureMap[0].value[0][0][i] = input[i];// Copy input to layer 0
			//m_Layer[0].m_FeatureMap[0].value[0][1][i] = input[i];// Copy input to layer 0
		}
	}
	else
	{
		for(i = 0; i < m_Layer[0].m_FeatureSize.row * m_Layer[0].m_FeatureSize.col; i++)
		{
			m_Layer[0].m_FeatureMap[0].value[0][1][i] = input[i];// Copy input to layer 0
		}
	}
		
	// Forward propagation: Calculate values of neurons in each layer 
	for(i=1; i<m_nLayer; i++)
	{
		if(epoch%PRINT_RESULT_INTERVAL == 0)
		{
			#ifdef PRINT_CNN_FAST
				if (m_Layer[i].m_layerIndex == 1){ char const * a = "./result/output_cnnf_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_cnnF(ans2, ios::app); SaveFile_cnnF << step << "\t"; SaveFile_cnnF.close(); }
			#endif
			#ifdef PRINT_CNN_MID
				if (m_Layer[i].m_layerIndex == 2){ char const * a = "./result/output_cnnm_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_cnnM(ans2, ios::app); 	SaveFile_cnnM << step << "\t"; SaveFile_cnnM.close(); }
			#endif
			#ifdef PRINT_CNN_SLOW
				if (m_Layer[i].m_layerIndex == 3){ char const * a = "./result/output_cnns_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_cnnS(ans2, ios::app); SaveFile_cnnS << step << "\t"; SaveFile_cnnS.close(); }
			#endif
			#ifdef PRINT_RNN_CTX
				if (m_Layer[i].m_layerIndex == 4){ char const * a = "./result/output_rnnm_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_rnnCtx(ans2, ios::app); SaveFile_rnnCtx << step << "\t";  SaveFile_rnnCtx.close(); }
			#endif
			#ifdef PRINT_RNN_CTX_FAST
				if (m_Layer[i].m_layerIndex == 5){ char const * a = "./result/output_rnnf_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_rnnCtx_fast(ans2, ios::app); SaveFile_rnnCtx_fast << step << "\t"; SaveFile_rnnCtx_fast.close(); }
			#endif
		}
		///////////////////////////////////////////////////////////////////////
		// Main Code: Forward Dynamics
		///////////////////////////////////////////////////////////////////////
		int batchIdxx = 0;
		m_Layer[i].Calculate(seq, step, epoch, batchIdxx,classNumber); 
		
		///////////////////////////////////////////////////////////////////////
		if(epoch%PRINT_RESULT_INTERVAL == 0)
		{
			#ifdef PRINT_CNN_FAST
				if (m_Layer[i].m_layerIndex == 1){ char const * a = "./result/output_cnnf_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_cnnF(ans2, ios::app); SaveFile_cnnF << "\n"; SaveFile_cnnF.close(); }
			#endif
			#ifdef PRINT_CNN_MID
				if (m_Layer[i].m_layerIndex == 2){ char const * a = "./result/output_cnnm_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_cnnM(ans2, ios::app); SaveFile_cnnM << "\n"; SaveFile_cnnM.close(); }
			#endif
			#ifdef PRINT_CNN_SLOW
				if (m_Layer[i].m_layerIndex == 3){ char const * a = "./result/output_cnns_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_cnnS(ans2, ios::app); SaveFile_cnnS << "\n"; SaveFile_cnnS.close(); }
			#endif
			#ifdef PRINT_RNN_CTX
				if (m_Layer[i].m_layerIndex == 4){ char const * a = "./result/output_rnnm_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_rnn(ans2, ios::app); SaveFile_rnn << "\n"; SaveFile_rnn.close(); }
			#endif
			#ifdef PRINT_RNN_CTX_FAST
				if (m_Layer[i].m_layerIndex == 5){ char const * a = "./result/output_rnnf_a"; int b = classNumber; stringstream sstm; sstm << a << b; string ans = sstm.str(); char * ans2 = (char *) ans.c_str(); ofstream SaveFile_rnn_Fast(ans2, ios::app); SaveFile_rnn_Fast << "\n"; SaveFile_rnn_Fast.close(); }
			#endif
		}
	}
	
}


//////////////////////////////////////////////////////////////////////////
void CCNN::Plot(int seq, int step, double xPos, double yPos, double angle, double xObs,int globalStep)
//////////////////////////////////////////////////////////////////////////
{
  if(0) // DO NOT SAVE ACTIVATION IMG
  {
	#ifdef USE_OPENCV
		int i;
		int interval = 50;
		int rowSize = 2 * interval;
		int colSize = (m_nLayer + 1) * interval;
		
		cv::Mat *L = new cv::Mat[m_nLayer];
		
		int rowMaxSize = 0;
		for(i = 0; i < m_nLayer; i++)
		{
			L[i] = m_Layer[i].Plot(seq, step);
			if(rowMaxSize < L[i].rows) rowMaxSize = L[i].rows;
			colSize += L[i].cols;
		}
		
		rowSize += rowMaxSize;
		
		cv::Mat image(rowSize, colSize, CV_8UC1, cv::Scalar::all(0));
		int x = interval, y = interval;
		for(i = 0; i < m_nLayer; i++)
		{
			cv::Mat imageROI = image(cv::Rect(x, y, L[i].cols, L[i].rows));
			L[i].copyTo(imageROI);
			
			x += L[i].cols + interval;
		}
		
		char path[200];
		//static int num = 0;
		//sprintf(path, "./result/outputActivation_xPos_%lf_yPos_%lf_rot_%d_obs_%lf_step_%03d.png", xPos,yPos,(int)angle,xObs,globalStep);
		//sprintf(path, "./result/outputActivation_%03d_%03d_%03d_%03d_step_%03d.png", (int)xPos,(int)yPos,(int)angle,(int)xObs,globalStep);
		sprintf(path, "./result/outputActivation_classNum_%03d_%03d.png", (int)xObs,globalStep);
		//sprintf(path, "./result/outputActivation_xPos_%lf_yPos_%lf_rot_%d_obs_%lf_step_%d.png", xPos,yPos,(int)angle,xObs,num++);
		//cv::imshow("CNN", image);
		cv::imwrite(path, image);
		cv::waitKey(1);	// 20 FPS
		
		
		delete[] L;
	#endif
  }
}


//////////////////////////////////////////////////////////////////////////
// Assign memory relate to calculate process (forward dynamics)
void CCNN::AllocWRTCal(int seqSize, int *stepSize, int numOfBatch)
//////////////////////////////////////////////////////////////////////////
{
	int idxLayer, idxFM;
	
	//assign memory relate to calculate process
	for(idxLayer = 0; idxLayer < m_nLayer; idxLayer++)
	{
		for(idxFM = 0; idxFM < m_Layer[idxLayer].m_nFeatureMap; idxFM++)
		{
			m_Layer[idxLayer].m_FeatureMap[idxFM].AllocWRTCal(seqSize, stepSize, numOfBatch);
		}
	}
}


//////////////////////////////////////////////////////////////////////////
void CCNN::DeleteWRTCal(int seqSize, int *stepSize)
//////////////////////////////////////////////////////////////////////////
{
	int idxLayer, idxFM;
	
	//free memory relate to calculate process
	for(idxLayer = 0; idxLayer <m_nLayer; idxLayer++){
		for(idxFM = 0; idxFM < m_Layer[idxLayer].m_nFeatureMap; idxFM++){
			m_Layer[idxLayer].m_FeatureMap[idxFM].DeleteWRTCal(seqSize, stepSize);
		}
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
void Layer::Construct(int type, int nFeatureMap, int FeatureSizeRow, int FeatureSizeCol, int KernelSizeRow, int KernelSizeCol, int SamplingFactor, double timeConst)
/////////////////////////////////////////////////////////////////////////////////////////////////////
{
	m_type = type;
	m_nFeatureMap = nFeatureMap;
	m_FeatureSize.row = FeatureSizeRow;
	m_FeatureSize.col = FeatureSizeCol;
	m_KernelSize.row = KernelSizeRow;
	m_KernelSize.col = KernelSizeCol;
	m_SamplingFactor = SamplingFactor;
	m_eta = 1.0 / timeConst;
	
	m_FeatureMap = new FeatureMap[ m_nFeatureMap ];

	for(int j=0; j<m_nFeatureMap; j++) 
	{
		m_FeatureMap[j].pLayer = this;
		m_FeatureMap[j].Construct();
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
void Layer::Construct_RNN(int type, int nFeatureMap, int FeatureSizeRow, int FeatureSizeCol, int KernelSizeRow, int KernelSizeCol, int SamplingFactor, double timeConst)
/////////////////////////////////////////////////////////////////////////////////////////////////////
{
	for(int j=0; j<m_nFeatureMap; j++) 
	{
		m_FeatureMap[j].Construct_RNN();
	}
}


/////////////////////////
void Layer::Delete()
/////////////////////////
{
	for(int j=0; j<m_nFeatureMap; j++)	m_FeatureMap[j].Delete();
}

///////////////////////////////////////////////////
// 1step, layer[-1] to layer[0] forward propagation
void Layer::Calculate(const int seq, const int step, const int epoch, const int batchIdx, const int classNumber)
///////////////////////////////////////////////////
{

	for(int i=0; i<m_nFeatureMap; i++)
	{
		///////////////////////////////////////////////////////////////
		// Forward Dynamics at RNN Output Layer
		///////////////////////////////////////////////////////////////
		if(m_type == RNN_OUTPUT){
				// Connection from the previous layer's (RNN_CTX) current step
				for(int j=0; j<pLayerPrev->m_nFeatureMap; j++)
				{
					m_FeatureMap[i].Calculate(
						pLayerPrev->m_FeatureMap[j].value[seq][step],		//input feature map
						j,													//index of input feature map
						seq, 
						step,
						PREVIOUS
							 );
				}
		}
		///////////////////////////////////////////////////////////////
		// Forward Dynamics at RNN CONTEXT Layer
		///////////////////////////////////////////////////////////////
		else if(m_type == MTRNN_FAST){
			// Initialize membrane potential (internal state)
			if(step == 0){
				for(int k=0; k < m_FeatureSize.row * m_FeatureSize.col; k++){
					m_FeatureMap[i].inter[seq][step][k] = 0.1; 
				}
			}
			else if(step > 0){
				// from the previous to the current
				for(int j=0; j<pLayerPrev->m_nFeatureMap; j++)
				{
						m_FeatureMap[i].Calculate(
							pLayerPrev->m_FeatureMap[j].value[seq][step-1],		//input feature map
							j,													//index of input feature map
							seq, 
							step,
							PREVIOUS
								 ); 
				}

				// from self
				if(step > 0){
					for(int j=0; j<m_nFeatureMap; j++)
					{
						m_FeatureMap[i].Calculate(
							m_FeatureMap[j].value[seq][step-1],		//input feature map
							j,													//index of input feature map
							seq, 
							step,
							CURRENT
								 ); 
					}
				}
			}
		}
		///////////////////////////////////////////////////////////////
		// Forward Dynamics at RNN CONTEXT BOUNDARY Layer
		///////////////////////////////////////////////////////////////
		else if(m_type == MTRNN_SLOW){
			// from the previous to the current
#ifdef CLAMPING
			if(i < NUM_NEURON_NOT_CONNECTED) 
			{} // Not connected.
			else
			{
				for(int j=0; j<pLayerPrev->m_nFeatureMap; j++)
				{
						m_FeatureMap[i].Calculate(
							pLayerPrev->m_FeatureMap[j].value[seq][step],		//input feature map
							j,//tempIdx,													//index of input feature map
							seq, 
							step,
							PREVIOUS
								 ); 
				}
			}
#else
			for(int j=0; j<pLayerPrev->m_nFeatureMap; j++)
			{
					m_FeatureMap[i].Calculate(
						pLayerPrev->m_FeatureMap[j].value[seq][step],		//input feature map
						j,													//index of input feature map
						seq, 
						step,
						PREVIOUS
							 ); 
			}
#endif
			if(step > 0)
			{
				// from self
				for(int j=0; j<m_nFeatureMap; j++)
				{
					m_FeatureMap[i].Calculate(
						m_FeatureMap[j].value[seq][step-1],		//input feature map
						j,													//index of input feature map
						seq, 
						step,
						CURRENT
							 ); 
				 }
				// from next
				for(int j=0; j<pLayerNext->m_nFeatureMap; j++)
				{
					m_FeatureMap[i].Calculate(
						pLayerNext->m_FeatureMap[j].value[seq][step-1],		//input feature map
						j,													//index of input feature map
						seq, 
						step,
						NEXT
							 ); 
				 }
			}
				
		}
		///////////////////////////////////////////////////////////////
		// Forward Dynamics at CNN Layers
		///////////////////////////////////////////////////////////////
		// Forward Dynamics at PFC Layer
		else if(m_type == PFC)
		{
			// Calculate the membrane potential w.r.t the previous layer (MSTNN SLOW)
			for(int j=0; j<pLayerPrev->m_nFeatureMap; j++)
			{
				m_FeatureMap[i].Calculate(
					pLayerPrev->m_FeatureMap[j].value[seq][step],		//input feature map
					j,													//index of input feature map
					seq, 
					step,
					PREVIOUS
						 );
			}
			if(step > 0)
			{
				// from self
				double summ = 0;				
				for(int j=0; j<m_nFeatureMap; j++)
				{
					summ += m_FeatureMap[j].value[seq][step-1][0] * m_FeatureMap[i].kernel_self[j][0];
				}
				m_FeatureMap[i].inter[seq][step][0] += summ;
				summ = 0;
				// from next
				for(int j=0; j<pLayerNext->m_nFeatureMap; j++)
				{
					summ += pLayerNext->m_FeatureMap[j].value[seq][step-1][0] * m_FeatureMap[i].kernel_next[j][0];
				}
				m_FeatureMap[i].inter[seq][step][0] += summ;
			}
			

		}		
		
		else{
			for(int j=0; j<pLayerPrev->m_nFeatureMap; j++)
			{
				m_FeatureMap[i].Calculate(
					pLayerPrev->m_FeatureMap[j].value[seq][step],		//input feature map
					j,													//index of input feature map
					seq, 
					step,
					PREVIOUS
						 );
			}
			
			
		}


		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Integration of each term to calculate the internal state (membrane potential)
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for(int k=0; k < m_FeatureSize.row * m_FeatureSize.col; k++)
		{
			m_FeatureMap[i].inter[seq][step][k] += m_FeatureMap[i].bias;
			m_FeatureMap[i].inter[seq][step][k] *= m_eta;
				
			// Previous its own internal state
			if(step > 0)
				m_FeatureMap[i].inter[seq][step][k] += (1 - m_eta) * m_FeatureMap[i].inter[seq][step - 1][k];
		}
		// Now we have the membrane potentials (inter[seq][step][k])

		///////////////////////////////////////////////////////////////
		// Calculate the activation value 
		///////////////////////////////////////////////////////////////
		if(m_type != RNN_OUTPUT)
		{
			for(int j=0; j < m_FeatureSize.row * m_FeatureSize.col; j++)
			{
				m_FeatureMap[i].value[seq][step][j] = activation(m_FeatureMap[i].inter[seq][step][j]);
			}
		}

		
		///////////////////////////////////////////////////////////////
		// Print out the value (Activation) of CNN_S (for debugging)
		///////////////////////////////////////////////////////////////
#ifdef TRAINING
#else
		if(epoch%PRINT_RESULT_INTERVAL == 0)
		{
			#ifdef PRINT_CNN_FAST
				if(m_layerIndex == 1){
					for(int j=0; j < m_FeatureSize.row * m_FeatureSize.col; j++)
					{
						char const * a = "./result/output_cnnf_a"; 
						int b = classNumber; //int b = batchIdx + 1;
						stringstream sstm;
						sstm << a << b;
						string ans = sstm.str();
						char * ans2 = (char *) ans.c_str();
						ofstream SaveFile_cnnF(ans2, ios::app); SaveFile_cnnF.precision(6); SaveFile_cnnF<<std::fixed;
						//ofstream SaveFile_cnnF("./result/log_ctx_cnnF.txt", ios::app); SaveFile_cnnF.precision(6); SaveFile_cnnF<<std::fixed;
						SaveFile_cnnF << m_FeatureMap[i].value[seq][step][j] << "\t"; SaveFile_cnnF.close();
					}
				}
			#endif
			#ifdef PRINT_CNN_MID
				if(m_layerIndex == 2){
					for(int j=0; j < m_FeatureSize.row * m_FeatureSize.col; j++)
					{
						char const * a = "./result/output_cnnm_a"; 
						int b = classNumber; //int b = batchIdx + 1;
						stringstream sstm;
						sstm << a << b;
						string ans = sstm.str();
						char * ans2 = (char *) ans.c_str();
						ofstream SaveFile_cnnM(ans2, ios::app); SaveFile_cnnM.precision(6); SaveFile_cnnM<<std::fixed;
						//ofstream SaveFile_cnnM("./result/log_ctx_cnnM.txt", ios::app); SaveFile_cnnM.precision(6); SaveFile_cnnM<<std::fixed;
						SaveFile_cnnM << m_FeatureMap[i].value[seq][step][j] << "\t"; SaveFile_cnnM.close();
					}
				}
			#endif
			#ifdef PRINT_CNN_SLOW
				if(m_layerIndex == 3){
					for(int j=0; j < m_FeatureSize.row * m_FeatureSize.col; j++)
					{
						char const * a = "./result/output_cnns_a"; 
						int b = classNumber; //int b = batchIdx + 1;
						stringstream sstm;
						sstm << a << b;
						string ans = sstm.str();
						char * ans2 = (char *) ans.c_str();
						ofstream SaveFile_cnnS(ans2, ios::app); SaveFile_cnnS.precision(6); SaveFile_cnnS<<std::fixed;
						//ofstream SaveFile_cnnS("./result/log_ctx_cnnS.txt", ios::app); SaveFile_cnnS.precision(6); SaveFile_cnnS<<std::fixed;
						SaveFile_cnnS << m_FeatureMap[i].value[seq][step][j] << "\t"; SaveFile_cnnS.close();
						//if(step == SET_CTX_STEP)
						//{
						//	cout << classNumber << "\t" << m_FeatureMap[i].value[seq][step][j] << "\n"; //getchar();
						//}
						
					}
				}
			#endif
			#ifdef PRINT_RNN_CTX
				if(m_layerIndex == 4){
					for(int j=0; j < m_FeatureSize.row * m_FeatureSize.col; j++)
					{
						char const * a = "./result/output_rnnm_a"; 
						int b = classNumber; //int b = batchIdx + 1;
						stringstream sstm;
						sstm << a << b;
						string ans = sstm.str();
						char * ans2 = (char *) ans.c_str();
						ofstream SaveFile_rnnC(ans2, ios::app); SaveFile_rnnC.precision(6); SaveFile_rnnC<<std::fixed;
						//ofstream SaveFile_rnnC("./result/log_ctx_rnn.txt", ios::app); SaveFile_rnnC.precision(6); SaveFile_rnnC<<std::fixed;
						SaveFile_rnnC << m_FeatureMap[i].value[seq][step][j] << "\t"; SaveFile_rnnC.close();
					}
				}
			#endif
			#ifdef PRINT_RNN_CTX_FAST
				if(m_layerIndex == 5){
					for(int j=0; j < m_FeatureSize.row * m_FeatureSize.col; j++)
					{
						char const * a = "./result/output_rnnf_a"; 
						int b = classNumber; //int b = batchIdx + 1;
						stringstream sstm;
						sstm << a << b;
						string ans = sstm.str();
						char * ans2 = (char *) ans.c_str();
						ofstream SaveFile_rnnC_fast(ans2, ios::app); SaveFile_rnnC_fast.precision(6); SaveFile_rnnC_fast<<std::fixed;
						//ofstream SaveFile_rnnC_fast("./result/log_ctx_rnn_fast.txt", ios::app); SaveFile_rnnC_fast.precision(6); SaveFile_rnnC_fast<<std::fixed;
						SaveFile_rnnC_fast << m_FeatureMap[i].value[seq][step][j] << "\t"; SaveFile_rnnC_fast.close();
					}
				}
			#endif
		}
#endif		
		
	}
	
#ifdef USE_SOFTMAX_OUTPUT		
	//SOFTMAX function
	if(m_type == OUTPUT_LAYER || m_type == RNN_OUTPUT){
		for(int idxOutput = 0 ; idxOutput < OUTPUT_DIMENSION ; idxOutput++)
		{
			double sum = 0;
			for(int i=idxOutput * SOFTMAX_DIMENSION; i<(idxOutput*SOFTMAX_DIMENSION)+SOFTMAX_DIMENSION; i++){
				sum += exp(m_FeatureMap[i].inter[seq][step][0]);
			}
			for(int i=idxOutput * SOFTMAX_DIMENSION; i<(idxOutput*SOFTMAX_DIMENSION)+SOFTMAX_DIMENSION; i++){
				m_FeatureMap[i].value[seq][step][0] = exp(m_FeatureMap[i].inter[seq][step][0]) / sum;
			}
		}
	}	
#else
	if(m_type == OUTPUT_LAYER || m_type == RNN_OUTPUT){
		for(int i=0; i<m_nFeatureMap; i++){
			m_FeatureMap[i].value[seq][step][0] = activation(m_FeatureMap[i].inter[seq][step][0]);
			
		}
	}
#endif



		


}

#ifdef USE_OPENCV
///////////////////////////////////////////////////
cv::Mat Layer::Plot(int seq, int step)		 
///////////////////////////////////////////////////
{
	int interval = 0;
	if(m_type == CONVOLUTIONAL)	interval = 2;
	
	int rowSize = interval * (m_nFeatureMap - 1);
	int colSize = 0;
	
	cv::Mat *L = new cv::Mat[m_nFeatureMap];
	
	//get image of the each feature map
	for(int i = 0; i < m_nFeatureMap; i++)
	{
		L[i] = m_FeatureMap[i].Plot(seq, step, m_FeatureSize, m_type);
	}
	
	rowSize += L[0].rows * m_nFeatureMap;
	colSize = L[0].cols;
	
	int nFMcol = m_nFeatureMap;		//number of feature map in each column
	int cntAddFm = 0;				//count additional feature map
	
	while(rowSize > 1000){		//too big to display -> divide into proper size
		if(nFMcol % 2 == 1){	//if nFMcol is odd
			cntAddFm++;
		}
		
		nFMcol /= 2;
		rowSize = L[0].rows * nFMcol + interval * (nFMcol - 1);
		colSize = 2 * colSize + interval; 
		
	}

	colSize += (L[0].cols + interval) * (int)ceil( (double)cntAddFm / nFMcol);
	
	cv::Mat image(rowSize, colSize, CV_8UC1, cv::Scalar::all(0));
	
	int cntFM = 0;
	int x = 0, y = 0;
	
	//combine whole feature maps into one
	while(cntFM < m_nFeatureMap)
	{
		//cv::Mat imageROI(image, cv::Range(y, y + L[cntFM].rows), cv::Range(x, x + L[cntFM].cols) );
		cv::Mat imageROI = image( cv::Rect(x, y, L[cntFM].cols, L[cntFM].rows) );
		L[cntFM].copyTo(imageROI);
		
		y += L[cntFM].rows + interval;	//move ROI position
		
		cntFM++;
		
		if(cntFM % nFMcol == 0){
			y = 0;
			x += L[0].cols + interval;
		}
	}

	//cv::imshow("temp", image);
	//cv::waitKey(0);
	
	delete[] L;
	
	return image;
}
#endif


//////////////////////////////////////////////////////////////////////////////////////////
// Construct CNN & Also RNN part (connection from the previous layer to the current layer)
void FeatureMap::Construct()
//////////////////////////////////////////////////////////////////////////////////////////
{
	if(pLayer->m_type == INPUT_LAYER) 
		m_nFeatureMapPrev = 0;
	else 
		m_nFeatureMapPrev = pLayer->pLayerPrev->m_nFeatureMap;

	MatSize KernelSize  = pLayer->m_KernelSize;

		
	// weights kernel
	kernel = new double* [ m_nFeatureMapPrev ];
	// Weight b/w previous and current layers
	for(int i=0; i<m_nFeatureMapPrev; i++) // i refers to the number of FM in the previous layer
	{
		kernel[i] = new double [KernelSize.row * KernelSize.col];
		// initialize bias
		bias = 0.05 * gaussianRand();

		
#ifdef INITIALIZE_METHOD_1		
		// initialize kernel (weight) based on normalized init. (Ref: Understanding the difﬁculty of training deep feedforward neural networks)
		if(m_nFeatureMapPrev != 0)
		{
			double begin = -1 * sqrt(6.0) / sqrt(pLayer->m_nFeatureMap + m_nFeatureMapPrev);
			double end = -1.0 * begin;
			for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel[i][j] = uniformRand(begin,end);
		}
		else
		{
			for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel[i][j] = 0.5 * gaussianRand();
		}
#endif
#ifdef INITIALIZE_METHOD_2		
		// initialize kernel (weight) based on normalized init. (Ref: Emergence of Functional Hierarchy in a Multiple Timescale Neural Network Model: A Humanoid Robot Experiment)
		double begin = -0.025; // -0.025;
		double end = 0.025; //0.025;
		for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel[i][j] = uniformRand(begin,end);
#endif
#ifdef INITIALIZE_METHOD_3		
		// initialize kernel (weight) based on normalized init.
		for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel[i][j] = 0.5 * gaussianRand();
#endif


		
	}
}


//////////////////////////////////////////////////////////////////////////////////////////
void FeatureMap::Construct_RNN()
//////////////////////////////////////////////////////////////////////////////////////////
{
	MatSize KernelSize  = pLayer->m_KernelSize;
	m_nFeatureMapNext = 0;

	
	// weights kernel_self (self: t-1 to t): Within the same layer
	kernel_self = new double* [ pLayer->m_nFeatureMap];
	for(int i=0; i<pLayer->m_nFeatureMap; i++) 
	{
		//kernel_self[i] = new double [KernelSize.row * KernelSize.col];
		kernel_self[i] = new double [1];
		
#ifdef MTRNN_KERNEL_INITIALIZE_METHOD_1				
		if(pLayer->m_nFeatureMap != 0 )
		{
			double begin = -1 * sqrt(6.0) / sqrt(pLayer->m_nFeatureMap + pLayer->m_nFeatureMap);
			double end = -1.0 * begin;
			for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel_self[i][j] = uniformRand(begin,end);
		}
		else
		{
			for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel_self[i][j] = 0.05 * gaussianRand();
		}
#endif
#ifdef MTRNN_KERNEL_INITIALIZE_METHOD_2		
		// initialize kernel_prev (weight) based on normalized init. (Ref: Emergence of Functional Hierarchy in a Multiple Timescale Neural Network Model: A Humanoid Robot Experiment)
		double begin = -0.025; // -0.025;
		double end = 0.025; //0.025;
		//for(int j=0; j < KernelSize.row * KernelSize.col; j++)
		for(int j=0; j < 1; j++)
		{
			kernel_self[i][j] = uniformRand(begin,end);
		}
#endif
#ifdef MTRNN_KERNEL_INITIALIZE_METHOD_3		// Added in this PFC Version
		for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel_self[i][j] = 0.5 * gaussianRand();
#endif

	}

	if(pLayer->m_type == MTRNN_SLOW || pLayer->m_type == PFC)
	{
		m_nFeatureMapNext = pLayer->pLayerNext->m_nFeatureMap;
		// Weight b/w the current layer and the next layer
		kernel_next = new double* [ m_nFeatureMapNext ];
		for(int i=0; i<m_nFeatureMapNext; i++)
		{
			//kernel_next[i] = new double [KernelSize.row * KernelSize.col];
			kernel_next[i] = new double [1];
#ifdef MTRNN_KERNEL_INITIALIZE_METHOD_1				
		if(pLayer->m_nFeatureMap != 0 )
		{
			double begin = -1 * sqrt(6.0) / sqrt(pLayer->m_nFeatureMap + pLayer->m_nFeatureMap);
			double end = -1.0 * begin;
			for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel_next[i][j] = uniformRand(begin,end);
		}
		else
		{
			for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel_next[i][j] = 0.05 * gaussianRand();
		}
#endif
#ifdef MTRNN_KERNEL_INITIALIZE_METHOD_2		
		// initialize kernel_prev (weight) based on normalized init. (Ref: Emergence of Functional Hierarchy in a Multiple Timescale Neural Network Model: A Humanoid Robot Experiment)
		double begin = -0.025; // -0.025;
		double end = 0.025; //0.025;
		//for(int j=0; j < KernelSize.row * KernelSize.col; j++)
		for(int j=0; j < 1; j++)
		{
			kernel_next[i][j] = uniformRand(begin,end);
		}
#endif
#ifdef MTRNN_KERNEL_INITIALIZE_METHOD_3		// Added in this PFC Version
		for(int j=0; j < KernelSize.row * KernelSize.col; j++) kernel_next[i][j] = 0.5 * gaussianRand();
#endif
			
		}
	}


	
}


///////////////////////////
void FeatureMap::Delete()
///////////////////////////
{
	for(int i=0; i<m_nFeatureMapPrev; i++) 
	{
		delete[] kernel[i];
	}
	delete[] kernel;
	
	if(pLayer->m_type == RNN_CTX_BND)
	{
		for(int i=0; i<pLayer->pLayerNext->m_nFeatureMap; i++) 
		{
			delete[] kernel_next[i];
		}
		delete[] kernel_next;
	}
	
	if(pLayer->m_type == RNN_CTX_BND || pLayer->m_type == RNN_CONTEXT)
	{
		for(int i=0; i<pLayer->m_nFeatureMap; i++) 
		{
			delete[] kernel_self[i];
		}
		delete[] kernel_self;
	}

}

//////////////////////////////////////////////////////////////////////////////////////////
void FeatureMap::AllocWRTCal(int seqSize, int *stepSize, int numOfBatch)
//////////////////////////////////////////////////////////////////////////////////////////
{
	MatSize FeatureSize = pLayer->m_FeatureSize;
	
	inter = new double**[seqSize];
	value = new double**[seqSize];
	
	for(int seq = 0; seq < seqSize; seq++){
		inter[seq] = new double*[stepSize[seq]];
		value[seq] = new double*[stepSize[seq]];
		for(int step = 0; step < stepSize[seq]; step++)
		{
			inter[seq][step] = new double[FeatureSize.row * FeatureSize.col];
			value[seq][step] = new double[FeatureSize.row * FeatureSize.col];
		}
	}

	// In order to match the learning between CNN and RNN, we use this trick.
	// CNN Slow value is saved, and it is updated every N epochs. ==> Updating parameters: CNN = every N epoch, RNN = every epoch
	if(pLayer->m_type == FULLY_CONNECTED)
	{
		int totalNumOfSeq = seqSize * numOfBatch;
		cnnSlowValue = new double**[totalNumOfSeq];
		cnnSlowValue_testing = new double**[totalNumOfSeq];
		
		for(int batch = 0 ; batch < numOfBatch; batch++)
		{
			cnnSlowValue[batch] = new double*[seqSize];
			cnnSlowValue_testing[batch] = new double*[seqSize];
			for(int seq = 0; seq < seqSize ; seq++)
			{
				cnnSlowValue[batch][seq] = new double[stepSize[seq]];
				cnnSlowValue_testing[batch][seq] = new double[stepSize[seq]];
			}
			
		}
			
	}

}

//////////////////////////////////////////////////////////////////////////////////////////
void FeatureMap::DeleteWRTCal(int seqSize, int *stepSize)
//////////////////////////////////////////////////////////////////////////////////////////
{
	for(int seq = 0; seq < seqSize; seq++){
		for(int step = 0; step < stepSize[seq]; step++){
			delete[] inter[seq][step];
			delete[] value[seq][step];
		}
		delete[] inter[seq];
		delete[] value[seq];
	}
	delete[] inter;
	delete[] value;
}

////////////////////////////
// Clear the internal state (membrane potential) as well as value?! This is initialization.
void FeatureMap::Clear(const int seqSize, const int *stepSize)
/////////////////////////////
{
	for(int seq = 0; seq < seqSize; seq++)
	{
		for(int step = 0; step < 2; step++)
		{
			for(int i=0; i < pLayer->m_FeatureSize.row * pLayer->m_FeatureSize.col; i++)
			{
				inter[seq][step][i] = 0.0;
				value[seq][step][i] = 0.0;
			}
		}
	}
}

#ifdef USE_OPENCV
//////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat FeatureMap::Plot(int seq, int step, MatSize mapSize, int type)
//////////////////////////////////////////////////////////////////////////////////////////////////////
{
	int scaling;

	if(type == INPUT_LAYER) 		scaling = 5;
	else if(type == CONVOLUTIONAL)	scaling = 5;
	else							scaling = 15;

	int rowSize = mapSize.row;
	int colSize = mapSize.col;

	int k = 0;
	
	cv::Mat L(rowSize * scaling, colSize * scaling, CV_8UC1);

	for(int r = 0; r < rowSize; r++){
		for(int c = 0; c < colSize; c++){
			for(int sr = 0; sr < scaling; sr++){
				for(int sc = 0; sc < scaling; sc++){
					if(type == INPUT_LAYER || type == OUTPUT_LAYER || type == RNN_OUTPUT) 
						//L.at<uchar>(r*scaling + sr, c*scaling + sc) = (unsigned char)( 255 * value[seq][step][k] );	//range : 0 - 1
						L.at<uchar>(r*scaling + sr, c*scaling + sc) = (unsigned char)( 255 * ((value[seq][step][k] / 2) + 0.5) );	//range : 0 - 1
						//L.at<uchar>(r*scaling + sr, c*scaling + sc) = (unsigned char)( 255 * value[seq][step][k] );	//range : 0 - 1
					else
						L.at<uchar>(r*scaling + sr, c*scaling + sc) = (unsigned char)( (value[seq][step][k] + 1.7159) * 255 / (2 * 1.7159) );	//range : -1.7159 - 1.7159
				}
			}
			
			k++;
		}
	}
	
	return L;
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////
//	calculate effect of a feature map in SOURCE LAYER (valueFeatureMapPrev) on this feature map in this layer
void FeatureMap::Calculate(double *valueFeatureMapPrev, const int &idxFeatureMapPrev, const int &seq, const int &step, const int &SOURCE)
//////////////////////////////////////////////////////////////////////////////////////////////////////
//	valueFeatureMapPrev:	feature map in SOURCE layer (Used as an input to this FM)
//	idxFeatureMapPrev :		index of feature map in SOURCE layer
{
	MatSize isize; //feature size in SOURCE layer
	MatSize ksize = pLayer->m_KernelSize; 
	unsigned int step_size = pLayer->m_SamplingFactor;
	
	if(SOURCE == PREVIOUS) 
		isize = pLayer->pLayerPrev->m_FeatureSize; 
	else if(SOURCE == CURRENT)
		isize = pLayer->m_FeatureSize;
	else //else if(SOURCE == NEXT)
		isize = pLayer->pLayerNext->m_FeatureSize; 
	
	int k = 0;
	
	// if kernel size = 1,1, feature size = 1,1, then loop once.
	for(unsigned int row0 = 0; row0 <= isize.row - ksize.row; row0 += step_size){ 
		for(unsigned int col0 = 0; col0 <= isize.col - ksize.col; col0 += step_size){
			// From the entire FMs in the SOURCE layer.
			// Convolute: Convolution, if the prevFMSize = 1 and Kernel Size = 1.. Then, it is Fully Connected
			if(SOURCE == PREVIOUS)
				inter[seq][step][k] += Convolute(valueFeatureMapPrev, isize, row0, col0, kernel[idxFeatureMapPrev], ksize);
			else if(SOURCE == CURRENT)
				inter[seq][step][k] += Convolute(valueFeatureMapPrev, isize, row0, col0, kernel_self[idxFeatureMapPrev], ksize);
			else if(SOURCE == NEXT)
				inter[seq][step][k] += Convolute(valueFeatureMapPrev, isize, row0, col0, kernel_next[idxFeatureMapPrev], ksize);
			k++;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//double FeatureMap::Convolute(double *input, const MatSize &size, const int r0, const int c0, double *weight, const MatSize &kernel_size)
double FeatureMap::Convolute(double *input, const MatSize &size, const unsigned int &r0, const unsigned int &c0, double *weight, const MatSize &kernel_size)
//////////////////////////////////////////////////////////////////////////////////////////////////////
{
	unsigned int i, j, k = 0;
	double summ = 0;
	
	for(i = r0; i < r0 + kernel_size.row; i++)
		for(j = c0; j < c0 + kernel_size.col; j++)
			summ += input[i * size.col + j] * weight[k++];

	return summ;
}


