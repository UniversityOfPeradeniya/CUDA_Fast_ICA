#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>    /* For O_RDWR */
#include <unistd.h>   /* For open(), creat() */
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <assert.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cula_lapack.h>
#include <cula_lapack_device.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <iomanip>

#include "lib/helpers.cuh"
#include "common.h"
#include "lib/myTimer.h"
#include "cudapart.cuh"

using namespace Eigen;
using namespace std;
using Eigen::ArrayXXd ;
using Eigen::MatrixXd ;


#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
cublasHandle_t handle;

void setGPU(){
	
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	if(nDevices==0){
		cout<<"No CUDA device found. Use the CPU version"<<endl;
		exit(1);
	}
	cudaDeviceProp prop;
	cout<<"Available devices : ";
	for(int i=0;i<nDevices;i++){
		cudaGetDeviceProperties(&prop, i);
		cout << prop.name << " ";
	}
	cout<<endl;

	int device=0;
	cudaSetDevice(device);checkCudaError();
	cudaDeviceReset(); checkCudaError()
	
	//to test if correctly set. Otherwise cublas change it_num
	int *a;
	cudaMalloc(&a,sizeof(int));checkCudaError()
	cudaFree(a);checkCudaError();
	
	device=-1;
	cudaGetDevice(&device); checkCudaError();
	cudaGetDeviceProperties(&prop, device);
	cout<<"Using GPU : "<<prop.name<<endl;
}

void endf(){
	int device=-1;
	cudaDeviceProp prop;
	cudaGetDevice(&device); 
	cudaGetDeviceProperties(&prop, device);
	cout<<"Used GPU : "<<prop.name<<endl;
}

/***************CULA error check**********************************************/
void checkStatus(culaStatus status)
{
    char buf[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    exit(EXIT_FAILURE);
}

/*BLAS functions helpers*/

void create_blas_handlers(){
	// Create a handle for CUBLAS
	cublasCreate(&handle);
	//cula
	culaStatus status = culaInitialize();
	checkStatus(status);
}

void destroy_blas_handlers(){
	 // Destroy the handle
	cublasDestroy(handle);
	culaShutdown();
}


/**********************/
/* cuBLAS ERROR CHECK */
/**********************/
#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if( CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err); 
        assert(0); 
    }
}







/*************************************COMMON FUNCTIONS****************************************/

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(double *A, double *B, double *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	
	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cudaDeviceSynchronize();checkCudaError();
	
}


void TransposeMatrixInCUBLAS(double *dv_ptr_in,double * dv_ptr_out,int m,int n){
	//m=number of rows
	//n = number of columns
	double alpha = 1.;
    double beta  = 0.;
    cublasSafeCall(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, dv_ptr_in, n, &beta, dv_ptr_in, n, dv_ptr_out, m)); 
    
}


__global__
void memSetInCuda(double *d_singleArray,double num,int sizeofSingleArray){
	
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	if(x<sizeofSingleArray){
		
		d_singleArray[x] = num;
	}
	
}












/***************************************PREPROCESSING**************************************/

void initializePreprocessCuda(preprocessVariables* DevicePointers,int n,int p){
	cudaMalloc((void**)&(DevicePointers->d_X_trf), n*p*sizeof(double));checkCudaError();
	cudaMalloc((void**)&(DevicePointers->d_means), n*sizeof(double));checkCudaError();
	cudaMalloc((void**)&(DevicePointers->d_X), n*p*sizeof(double));checkCudaError();
	cudaMalloc((void**)&(DevicePointers->d_X_init), n*p*sizeof(double));checkCudaError();
	cudaMalloc((void**)&(DevicePointers->d_singleArray), p*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->d_S), imin(p,n)*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->d_VT), n*n*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->d_VTT), n*n*sizeof(double));checkCudaError();
	cudaMalloc((void**)&(DevicePointers->d_X1), n*p*sizeof(double));checkCudaError();
	cudaMalloc((void**)&(DevicePointers->d_X1_T), p*n*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->d_w_init_tr), n*n*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->d_w_init_w_init_tr), n*n*sizeof(double));checkCudaError();	
	cudaMalloc( (void**)&(DevicePointers->WR), n*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->WI), n*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->W), n*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->VL), n*n*sizeof(double));checkCudaError();  
	cudaMalloc( (void**)&(DevicePointers->output), n*n*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->d_eigenVectorT), n*n*sizeof(double));checkCudaError();
	cudaMalloc( (void**)&(DevicePointers->d_output_eigenVectorT), n*n*sizeof(double));checkCudaError();
}
	
	
void getMeanNormalizeOnCUDA(MatrixXd& X,int n,int p,preprocessVariables* DevicePointers ){
	
	int blockSize = 512;
    int gridSize = (int)ceil(((double)(p))/blockSize);
	double memSetNumber = 1.0/p;
	
	double * xpointer = X.data();
	
	dim3 blockSizeNorm(16,16);
	dim3 gridSizeNorm((int)ceil(((double)p)/blockSizeNorm.x),(int)ceil(((double)n)/blockSizeNorm.y));
    	
	cudaMemcpy( (DevicePointers->d_X), xpointer,  n*p*sizeof(double), cudaMemcpyHostToDevice );checkCudaError();
	cudaMemcpy( (DevicePointers->d_X_init), xpointer,  n*p*sizeof(double), cudaMemcpyHostToDevice );checkCudaError();

	memSetInCuda<<<gridSize, blockSize>>>((DevicePointers->d_singleArray),memSetNumber,p);
	gpu_blas_mmul((DevicePointers->d_X), (DevicePointers->d_singleArray), (DevicePointers->d_means),n,p,1);
	cudaDeviceSynchronize();checkCudaError();
	
	//calling kernal
    normalizeInCUDA<<<gridSizeNorm, blockSizeNorm>>>((DevicePointers->d_X),(DevicePointers->d_means),n,p);
    cudaDeviceSynchronize();checkCudaError();
	
	//Transpose matrix using CUBLAS
	TransposeMatrixInCUBLAS((DevicePointers->d_X),DevicePointers->d_X_trf,p,n);
	cudaDeviceSynchronize();checkCudaError();


}

__global__
void normalizeInCUDA(double * d_X,double * d_means,int n,int p){

	//should optimize with shared memory
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	
	if(y<n && x<p){
		int index;
		index = x*n+y;
		//printf("%d %d %lf %lf \n",index,index%n,d_X[index],d_means[index%n]);
		d_X[index] = d_X[index]- d_means[index%n];
		
	}	
}



/*SVD functions*/

int runSVDonCUDA(double * A,preprocessVariables* DevicePointers,int ROWS,int COLS){
	
	//initilaize matrix
	
	int M = ROWS;
    int N = COLS;

    double* U = NULL;
	
	culaStatus status;

	/* Setup SVD Parameters */
    int LDA = M;
    int LDU = M;
    int LDVT = N;

    char jobu = 'N';
    char jobvt = 'A';

	/* Initialize CULA */
    //status = culaInitialize();
	/* Perform singular value decomposition CULA */
    status = culaDeviceDgesvd(jobu, jobvt, M, N, A, LDA, (DevicePointers->d_S), U, LDU, (DevicePointers->d_VT), LDVT);
    checkStatus(status);
	//culaShutdown();
	cudaDeviceSynchronize();checkCudaError();

    return EXIT_SUCCESS;

}


void devideVTbySingularValues(double * d_VT,double * d_VTT, double * d_S,int n){
	
	dim3 blockSizeNorm(16,16);
	dim3 gridSizeNorm((int)ceil(((double)n)/blockSizeNorm.x),(int)ceil(((double)n)/blockSizeNorm.y));
	
	devideVTInCUDA<<<gridSizeNorm, blockSizeNorm>>>(d_VT,d_S,n);
	cudaDeviceSynchronize();checkCudaError();
	
	//Transpose d_VT to d_VTT
	TransposeMatrixInCUBLAS(d_VT,d_VTT,n,n);
	
}

__global__
void devideVTInCUDA(double *d_VT,double *d_S, int n){

	//should optimize with shared memory
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	
	if(y<n && x<n){
		int index;
		index = x*n+y;
		//printf("%d %d %lf %lf \n",index,index%n,d_VT[index],d_S[index%n]);
		d_VT[index] = d_VT[index] / d_S[index%n];
		
	}	
}


void multiplyOnGPU_K_X(preprocessVariables* DevicePointers,int n,int p){
	
	gpu_blas_mmul_X1(DevicePointers->d_VT, DevicePointers->d_X, DevicePointers->d_X1,n,n,p); //X1 matrix
	cudaDeviceSynchronize();checkCudaError();
	TransposeMatrixInCUBLAS(DevicePointers->d_X1,DevicePointers->d_X1_T,p,n); //X1 Transpose matrix
	
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul_X1(double *A, double *B, double *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const double alf = (double)sqrt((double)n);
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	
	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cudaDeviceSynchronize();checkCudaError();
	
}





/***************************************PARELLEL LOOP**************************************/

void memSetForSymDecorrelationCUDA(MatrixXd& w_init,preprocessVariables* DevicePointers,int n){
	
	
	double * datapointer = w_init.data();
	cudaMalloc( (void**)&(DevicePointers->d_w_init), n*n*sizeof(double));checkCudaError();	
	cudaMemcpy( DevicePointers->d_w_init, datapointer,n*n*sizeof(double), cudaMemcpyHostToDevice );checkCudaError();

}


timestamp_t mulTime2SUM=0,eigsTimeSUM=0,divideTimeSUM=0,mulTime3SUM=0;
timestamp_t tmptime;


timestamp_t getmulTime2SUM(){
	return mulTime2SUM;
}
timestamp_t geteigsTimeSUM(){
	return eigsTimeSUM;
}
timestamp_t getdivideTimeSUM(){
	return divideTimeSUM;
}
timestamp_t getmulTime3SUM(){
	return mulTime3SUM;
}




void sym_decorrelation_cuda(preprocessVariables* DevicePointers,int n){

	tmptime = get_timestamp();
	TransposeMatrixInCUBLAS(DevicePointers->d_w_init,(DevicePointers->d_w_init_tr),n,n);
	gpu_blas_mmul(DevicePointers->d_w_init,(DevicePointers->d_w_init_tr),(DevicePointers->d_w_init_w_init_tr),n,n,n);
	mulTime2SUM+=(get_timestamp() - tmptime) ;
	

	//find eigenvalues
	culaStatus status;

	//job parameters
    char JOBVL = 'V';
    char JOBVR = 'N';
	//int n is given
	double * A = DevicePointers->d_w_init_w_init_tr;
	
	/* Setup Parameters leading dimension*/
    int LDA = n;
	int LDVL = n;
	int LDVR = n;
	
    
	double * WR = DevicePointers->WR;
	double * WI = DevicePointers->WI; 
    double * W = DevicePointers->W;
	double * VL = DevicePointers->VL;
	double * VR = NULL;


	tmptime = get_timestamp();	
	// Initialize CULA
    //status = culaInitialize();
	/* SGEEV prototype */
	/*extern void sgeev( char* jobvl, char* jobvr, int* n, double* a,
                int* lda, double* wr, double* wi, double* vl, int* ldvl,
                double* vr, int* ldvr, double* work, int* lwork, int* info );*/
    status = culaDeviceDgeev(JOBVL,JOBVR, n, A, LDA, WR, WI,VL, LDVL,VR,LDVR);
    checkStatus(status);
	//culaShutdown();
	eigsTimeSUM+=(get_timestamp() - tmptime) ;
	
	DevicePointers->d_VTT=invokeDevideByDiagonal(DevicePointers,A,VL,DevicePointers->d_w_init,n);
	cudaDeviceSynchronize();checkCudaError();
}

double * invokeDevideByDiagonal(preprocessVariables* DevicePointers,double * d_eigenValues, double * d_eigenVectors,double * d_w_init,int n){
	
	dim3 blockSize(16,16);
	dim3 gridSize((int)ceil(((double)n)/blockSize.x),(int)ceil(((double)n)/blockSize.y));
	
	tmptime = get_timestamp();
	int ERROR = 0;
	//printf("Error value before %d\n",ERROR);
    devideByDiagonal<<<gridSize, blockSize>>>(d_eigenValues,d_eigenVectors,(DevicePointers->output),n,n,ERROR);
	cudaDeviceSynchronize();checkCudaError();
	divideTimeSUM+=(get_timestamp() - tmptime) ;

	tmptime = get_timestamp();
	TransposeMatrixInCUBLAS(d_eigenVectors,(DevicePointers->d_eigenVectorT),n,n);
	gpu_blas_mmul((DevicePointers->output),(DevicePointers->d_eigenVectorT),(DevicePointers->d_output_eigenVectorT),n,n,n);
	gpu_blas_mmul((DevicePointers->d_output_eigenVectorT),d_w_init,(DevicePointers->output),n,n,n);
	mulTime3SUM+=(get_timestamp() - tmptime) ;	
	
	return (DevicePointers->output);
	
}

__global__
void devideByDiagonal(double *d_eigenValues,double * d_eigenVectors,double * output,int n,int p, int ERROR){

	//should optimize with shared memory
	//n means row, p means columns
	//since this is square matrix, n=p
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	ERROR = 0;
	if(y<n && x<p){
		int index;
		index = x*n+y;
		//printf("%d %d %d %f %d %f\n",x,y,index,d_eigenVectors[index], (n*x)+x, 1/sqrt(d_eigenValues[(n*x)+x]));
		double val = d_eigenValues[(n*x)+x];
		// if(val<=0){
			// printf("\nI Got negative\n");
			// ERROR = 1;
			// printf("Now error value %d\n",ERROR);
		// }
		output[index] = d_eigenVectors[index]*(1/sqrt(val));
	}	
}


cudaVar initializeParallelCuda(preprocessVariables* DevicePointers,cudaVar cudaVariables,int n,int p){
	
	//matrix sizes
	const int matsizeW = n*n*sizeof(double);
	const int matsizeW1 = n*n*sizeof(double);
	const int matsizeProduct = n*p*sizeof(double);
	const int matsizegwtx = n*p*sizeof(double);
	const int matsizeCubeDerivation = n*p*sizeof(double);
	const int matsizeg_wtx = n*1*sizeof(double);
	const int matsizeGwtxIntoXtranspose = n*n*sizeof(double);
	const int matsizeGwtx_into_W = n*n*sizeof(double);
	const int matsizew_init_w_init_T = n*n*sizeof(double);
	const int matsizeEigenValues = n*1*sizeof(double);
	const int matsizeEigenVectors = n*n*sizeof(double);
	const int matsizeEigenRowWise = n*n*sizeof(double);
	
	const int matsizeW1intoWT = n*n*sizeof(double);
	
	const int matsizebw = n*1*sizeof(double);
	const int matsizezw = n*1*sizeof(double);
	
	const int matsizediagonal = n*1*sizeof(double);
	
	const int matsizeit_num = sizeof(int);
	const int matsizerot_num = sizeof(int);
	const int matsizeAnswer = sizeof(double);
	const int matsize_tmp_w_init = n*n*sizeof(double);

	const int sizeofSingleArray = p*1*sizeof(double);
	const int sizeofComputedArray = n*1*sizeof(double);
	
	//malloc
	cudaMalloc( (void**)&cudaVariables.W1, matsizeW1 );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.product, matsizeProduct );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.gwtx, matsizegwtx );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.cubeD, matsizeCubeDerivation );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.g_wtx, matsizeg_wtx );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.g_wtx_X1_transpose, matsizeGwtxIntoXtranspose );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.gwtx_into_W, matsizeGwtx_into_W );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.w_init_w_init_T, matsizew_init_w_init_T );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.eigenValues, matsizeEigenValues );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.eigenVectors, matsizeEigenVectors );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.eigenRowWise, matsizeEigenRowWise );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.W1intoWT, matsizeW1intoWT );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.diagonal, matsizediagonal );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.answer, matsizeAnswer );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.tmp_w_init, matsize_tmp_w_init );checkCudaError();
	
	//malloc
	cudaMalloc( (void**)&cudaVariables.bw, matsizebw );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.zw, matsizezw );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.it_num, matsizeit_num );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.rot_num, matsizerot_num );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.d_singleArray, sizeofSingleArray );checkCudaError();
	cudaMalloc( (void**)&cudaVariables.d_computeArray, sizeofComputedArray );checkCudaError();
	cudaMallocHost((void **) &cudaVariables.hostpointer,matsizeW);checkCudaError();
	
	
	//MemSet in CUDA
	int blockSize = 512;
    int gridSize = (int)ceil(((double)(p))/blockSize);
	memSetInCuda<<<gridSize, blockSize>>>(cudaVariables.d_singleArray,1.0,p);
	cudaDeviceSynchronize();checkCudaError();
	
	//copy data to CUDA
	cudaVariables.X1 = DevicePointers->d_X1;
	cudaVariables.X1Transpose = DevicePointers->d_X1_T;
	cudaVariables.W = DevicePointers->d_VTT;
	cudaVariables.w_init = DevicePointers->d_w_init;

	cudaDeviceSynchronize();
	checkCudaError();
	
	return cudaVariables;

}
	
void matrixMultiplyonGPU(double * d_A, double * d_B, double * d_C,int n,int p){	
	gpu_blas_mmul(d_A, d_B, d_C, n, n, p);
	cudaDeviceSynchronize();checkCudaError();
}	
	
	
	
timestamp_t cubeTimeSUM=0,mulTime=0,meanTime=0,cubeTime1,mulTime1,meanTime1;
timestamp_t getCubeTimeSUM(){
	return cubeTimeSUM;
}
timestamp_t getMulTime(){
	return mulTime;
}
timestamp_t getMeanTime(){
	return meanTime;
}
	
void cubeOnGPU(cudaVar cudaVariables,int n,int p){
	double * original=cudaVariables.product;
	double * computed=cudaVariables.gwtx;
	double * cubeDerivation=cudaVariables.cubeD;
	double * g_wtx = cudaVariables.g_wtx;
	
    dim3 blockSize(16,16);
	dim3 gridSize((int)ceil(((double)p)/blockSize.x),(int)ceil(((double)n)/blockSize.y));
	
	//g(WX) and g'(WX)
	cubeTime1 = get_timestamp();
    applyCube<<<gridSize, blockSize>>>(original,computed,cubeDerivation,n,p);
	cudaDeviceSynchronize();checkCudaError();
	cubeTimeSUM+=(get_timestamp() - cubeTime1) ;

	double * d_singleArray = cudaVariables.d_singleArray;
	double * d_computeArray = cudaVariables.d_computeArray;

	//mean of each row
	mulTime1 = get_timestamp();
	gpu_blas_mmul(cubeDerivation, d_singleArray, d_computeArray,n,p,1);
	mulTime+=(get_timestamp() - mulTime1) ;
	
	blockSize = n;
    gridSize = 1;
	//original mean function
    
	//divide by means to get E(g'(WX))
	meanTime1 = get_timestamp(); 
	findMean<<<gridSize, blockSize>>>(g_wtx,d_computeArray,n,p);
    cudaDeviceSynchronize();checkCudaError();
	meanTime+=(get_timestamp() - meanTime1);
	

	
}	

__global__ 
void applyCube(double *original,double * computed,double * cubeDerivation, int n,int p) 
{
	
	//converted to 2d
	
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	if(x<p && y<n){
		double originalFloat = original[y*p+x];
		double tmp = originalFloat*originalFloat;
		cubeDerivation[y*p+x]=3*tmp;
		computed[y*p+x] =tmp*originalFloat;
	}
	
}
	
__global__ 
void findMean(double * computed,double * alternative,int n,int p) 
{
	
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	
	computed[x]=alternative[x]/p;
	
}

	
	
void matrixMultiplyTransposeImprovedonGPU(cudaVar cudaVariables,double p_,int n,int p){
	double * d_A = cudaVariables.gwtx;
	double * d_B = cudaVariables.X1Transpose;
	double * d_C = cudaVariables.g_wtx_X1_transpose;
	
	gpu_blas_mmulImprove(d_A, d_B, d_C, n, p, n,p_);
	cudaDeviceSynchronize();checkCudaError();
	
}
	
// Improved version for transpose multiplication
void gpu_blas_mmulImprove(double *A, double *B, double *C, const int m, const int k, const int n,const double p_) {
	int lda=m,ldb=k,ldc=m;
	const double alf = 1/p_;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cudaDeviceSynchronize();checkCudaError();
}	
	
	
void multiplyColumnViseOnGPU(cudaVar cudaVariables,int n,int p){
	double * original=cudaVariables.W;
	double * calculated = cudaVariables.gwtx_into_W;
	double * factor=cudaVariables.g_wtx;
		
	dim3 blockSize(16,16);
	dim3 gridSize((int)ceil(((double)p)/blockSize.x),(int)ceil(((double)n)/blockSize.y));
    
    multiplyColumnVise<<<gridSize, blockSize>>>(original,calculated,factor,n,n);
    cudaDeviceSynchronize(); checkCudaError();
}
	
__global__ 
void multiplyColumnVise(double *original,double * calculated,double * factor,int n,int p) 
{
	
	//int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	//int i;
	
	if(y<n && x<p){
		double mulFactor;
		int index;
		
		mulFactor = factor[y];
		index = x*n+y;
		calculated[index] = original[index]*mulFactor;

	}	
}	
	
void subtractOnGPU(cudaVar cudaVariables,int n){
	double * gwtx_into_x1transpose_p=cudaVariables.g_wtx_X1_transpose;
	double * gwtx_into_W=cudaVariables.gwtx_into_W;
	double * target=cudaVariables.w_init;

	int blockSize, gridSize;
    blockSize = 512;
    gridSize = (int)ceil(((double)(n*n))/blockSize);

    subtractMatrices<<<gridSize, blockSize>>>(target,gwtx_into_x1transpose_p,gwtx_into_W,n);
    cudaDeviceSynchronize();checkCudaError();

}

__global__ 
void subtractMatrices(double * target,double *A,double * B,int n) 
{
	
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	if(x<n*n){
		target[x] = A[x]-B[x];
	}
		
}


void copyBackW1fromCUDA(MatrixXd& W1,double * d_W1,int n){
	MatrixXd tmp(n,n);
	cudaMemcpy(tmp.data(),d_W1,n*n*sizeof(double),cudaMemcpyDeviceToHost);
	W1 = tmp.cast<double>();
}








/***********************************************POSTPROCESSING*******************************/

void copytoHost(MatrixXd& K,double* d_VT,int n,int p){
	MatrixXd tmp(n,p);
	cudaMemcpy(tmp.data(),d_VT,n*p*sizeof(double),cudaMemcpyDeviceToHost);
	K = tmp;
}
// void copyResulttoHost(MatrixXd& K,preprocessVariables* DevicePointers,int n,int p){
	// MatrixXd tmp(p,n);
	// cudaMemcpy(tmp.data(),DevicePointers->d_X1_T,p*n*sizeof(double),cudaMemcpyDeviceToHost);
	// K = tmp;
// }


/*UNUSED*/

void saveW1inGPU(MatrixXd& W1,cudaVar cudaVariables,int n){
	
	MatrixXd f_W1 = W1.cast<double>();
	double *dataFromW1 = f_W1.data();
	const int matsizeW1 = n*n*sizeof(double);
	
	cudaMemcpy( cudaVariables.W, dataFromW1, matsizeW1, cudaMemcpyHostToDevice );
	cudaDeviceSynchronize();
	
}

// MatrixXd tmp(n,n);
// cudaMemcpy(tmp.data(),d_eigenValues,n*n*sizeof(double),cudaMemcpyDeviceToHost);
// cout<<"Eigenvalues"<<endl<<tmp<<endl;	


__global__
void castToFloatInCUDA(double * input,double * output,int n,int p){

	//should optimize with shared memory
	int y = blockIdx.y*blockDim.y+threadIdx.y; //n
	int x = blockIdx.x*blockDim.x+threadIdx.x; //p
	
	
	if(y<n && x<p){
		int index;
		index = x*n+y;
		//printf("%d %d %lf %lf \n",index,index%n,d_X[index],d_means[index%n]);
		output[index] = (double)input[index];
		
	}	
}
