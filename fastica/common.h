struct CUDA_variables{
	double * X1;
	double * X1Transpose;
	double * W;
	double * W1;
	double * w_init;
	double * product;
	double * gwtx;
	double * g_wtx;
	double * cubeD;
	double * g_wtx_X1_transpose;
	double * gwtx_into_W;
	double * w_init_w_init_T;
	double * eigenValues;
	double * eigenVectors;
	double * W1intoWT;
	
	int* it_num;
	int* rot_num;
	double *bw;
	double *zw;
	
	double * eigenRowWise;
	double * diagonal;
	double * answer;
	
	double * tmp_w_init;
	
	double * d_singleArray;
	double * d_computeArray;
	
	double * hostpointer;
	
	};

struct preprocessVariableList{
	double * d_X;
	double * d_X_init;
	double * d_Xf;
	double * d_X_trf;
	double * d_VT;
	double * d_VTT;
	double * d_S;
	double * d_w_init;
	double * d_w_init_T;
	double * d_X1;
	double * d_X1_T;
	double * d_means;
	double * d_singleArray;	
	double * d_w_init_tr;
	double * d_w_init_w_init_tr;	
	
	//used for sym_decorrelation_cuda
	double * WR;
	double * WI; 
    double * W;
	double * VL;	
	//invokeDevideByDiagonal
	double * output;
	double * d_eigenVectorT;
	double * d_output_eigenVectorT;
	
};

typedef struct CUDA_variables cudaVar;
typedef struct preprocessVariableList preprocessVariables;

