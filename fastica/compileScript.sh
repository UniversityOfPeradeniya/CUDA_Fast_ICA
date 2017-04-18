make clean
rm ../output/output.txt
make 2>&1 | grep error

nvcc -arch=sm_20 -m64 -o out cudapart.cu ./lib/helpers.cu main.o -I "../" -lcublas -lcula_lapack -I${CULA_INC_PATH} -L${CULA_LIB_PATH_64} 2>&1 | grep error

strip out
rm main.o
