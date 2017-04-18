
__global__ void normalizeInCUDA(double * d_X,double * d_means,int n,int p);
__global__ void devideVTInCUDA(double *d_VT,double *d_S, int n);
__global__ void devideByDiagonal(double *d_eigenValues,double * d_eigenVectors,double * output,int n,int p, int ERROR);
__global__ void applyCube(double *original,double * computed,double * cubeDerivation, int n,int p) ;
__global__ void findMean(double * computed,double * alternative,int n,int p) ;
__global__ void multiplyColumnVise(double *original,double * calculated,double * factor,int n,int p) ;
__global__ void subtractMatrices(double * target,double *A,double * B,int n) ;
	
void gpu_blas_mmul_X1(double *A, double *B, double *C, const int m, const int k, const int n);
double * invokeDevideByDiagonal(preprocessVariables* DevicePointers,double * d_eigenValues, double * d_eigenVectors,double * d_w_init,int n);
void gpu_blas_mmulImprove(double *A, double *B, double *C, const int m, const int k, const int n,const double p_);
	