ifeq ($(CXX),icpc)
CXX_FLAGS = -qopenmp -xhost -restrict 
else
ifeq ($(CXX),g++)
CXX_FLAGS = -fopenmp -march=native
endif
endif

CXX_FLAGS += -O3 -std=c++11 -Wall
CXX_LINK = ${BLAS_LIB}

ifeq (${MAKECMDGOALS},GPU2)
TYPE=cu 
else
TYPE= cpp
endif

CPP_TRANSPOSE=$(wildcard ttc_transpositions/*.${TYPE})
OBJ_TRANSPOSE=$(addprefix ./,$(notdir $(CPP_TRANSPOSE:.${TYPE}=.o)))

SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)

CPU: 
	${MAKE} clean 
	${MAKE} CPU2
CPU2: all

GPU:
	${MAKE} clean
	${MAKE} GPU2
GPU2: CXX_FLAGS = -rdc true -O3 -lgomp -Xcudafe "--diag_suppress=declared_but_not_referenced" ${CUDA_INCLUDE}
GPU2: CXX_LINK = ${CUDA_LIB}
GPU2: CXX = nvcc
GPU2: all

all: ${OBJ} ${CPP_TRANSPOSE}
	${CXX} ${OBJ} ${CPP_TRANSPOSE} ${CXX_FLAGS} -o gett.exe ${CXX_LINK}

%.o: %.cpp
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@ 
ttc_transpositions/%.o: ttc_transpositions/%.cu
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@ 

clean:
	rm -f *.o ${OBJ} ${OBJ_TRANSPOSE} gett.exe
