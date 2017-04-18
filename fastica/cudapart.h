void setGPU();
void endf();
cudaVar initializeParallelCuda(preprocessVariables* DevicePointers, cudaVar cudaVariables,int n,int p);
void initializePreprocessCuda(preprocessVariables* DevicePointers,int n,int p);
void matrixMultiplyonGPU(double * d_A, double * d_B, double * d_C,int n,int p);
void copyBackProductfromCUDA(MatrixXd& product,double * from);
void cubeOnGPU(cudaVar cudaVariables,int n,int p);
void copyBackCubefromCUDA(MatrixXd& cube,double * from);
void copyBackCubeDerivationfromCUDA(MatrixXd& cubeDerivation,double * from);
void copyBackX1fromCUDA(MatrixXd& X1,double * from);

void copyBackTransposeMulfromCUDA(MatrixXd& tr,double * from);
void multiplyColumnViseOnGPU(cudaVar cudaVariables,int n,int p);
void subtractOnGPU(cudaVar cudaVariables,int n);
void saveW1inGPU(MatrixXd& W1,cudaVar cudaVariables,int n);
void copyBackWfromCUDA(MatrixXd& W,double * from);
void matrixMultiplyTransposeImprovedonGPU(cudaVar cudaVariables,double p_,int n,int p);
void create_blas_handlers();
void destroy_blas_handlers();
cudaVar findEigenOnCuda(cudaVar cudaVariables);
int runSVDonCUDA(double * A,preprocessVariables* DevicePointers,int ROWS,int COLS);

void getMeanNormalizeOnCUDA(MatrixXd& X,int n,int p,preprocessVariables* DevicePointers );
void devideVTbySingularValues(double * d_VT,double * d_VTT, double * d_S,int n);
void multiplyOnGPU_K_X(preprocessVariables* DevicePointers,int n,int p);
void TransposeMatrixInCUBLAS(double *dv_ptr_in,double * dv_ptr_out,int m,int n);

void sym_decorrelation_cuda(preprocessVariables* DevicePointers,int n);
void memSetForSymDecorrelationCUDA(MatrixXd& w_init,preprocessVariables* DevicePointers,int n);
void copytoHost(MatrixXd& K,double* d_VT,int n,int p);

void copyBackW1fromCUDA(MatrixXd& W1,double * d_W1,int n);


typedef unsigned long long timestamp_t;
timestamp_t getCubeTimeSUM();
timestamp_t getMulTime();
timestamp_t getMeanTime();
timestamp_t getmulTime2SUM();
timestamp_t geteigsTimeSUM();
timestamp_t getdivideTimeSUM();
timestamp_t getmulTime3SUM();

