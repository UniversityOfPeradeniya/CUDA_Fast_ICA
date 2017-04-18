# You may need to edit this file to reflect the type and capabilities of your system.
# The defaults are for a Linux system and may need to be changed for other systems (eg. Mac OS X).


CXX=g++
INPUT = main.cpp 
INCLUDE_FLAG = -c -I "../"  -O3 -std=c++11 -Wl,--no-as-needed
CXXFLAGS = $(INCLUDE_FLAG) 
#ARCH=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_20,code=sm_20
ARCH=-arch=sm_35

out: main.o cudapart.cu ./lib/helpers.cu
	nvcc $(ARCH) -w -m64 -o out cudapart.cu ./lib/helpers.cu main.o -I "../" -lcublas -lcula_lapack -I${CULA_INC_PATH} -L${CULA_LIB_PATH_64}

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INPUT)  -o $@ 

.PHONY: clean

clean:
	rm -f out main.o out ../output/output.txt

