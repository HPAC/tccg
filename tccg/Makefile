BLAS_LIB=-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
CUDA_LIB=-L${CUDA_ROOT}/lib64 -lcublas
CUDA_INCLUDE=-I${CUDA_ROOT}/include
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
CPP_TRANSPOSE=$(wildcard ttc_transpositions/*.cu)
OBJ_TRANSPOSE=$(addprefix ./,$(notdir $(CPP_TRANSPOSE:.cu=.o)))
else
CPP_TRANSPOSE=$(wildcard ttc_transpositions/*.cpp)
OBJ_TRANSPOSE=$(addprefix ./,$(notdir $(CPP_TRANSPOSE:.cpp=.o)))
endif


SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)

CPU: 
	${MAKE} clean 
	${MAKE} CPU2
CPU2: INCLUDE_PATH=-I${HPTT_ROOT}/include/ -I${TCL_ROOT}/include/
CPU2: CXX_FLAGS+=-L${HPTT_ROOT}/lib -lhptt ${TCL_ROOT}/lib/libtcl.so
CPU2: all

GPU:
	${MAKE} clean
	${MAKE} GPU2
GPU2: CXX_FLAGS = ${CUDA_ARCH} -rdc true -O3 -lgomp -Xcudafe "--diag_suppress=declared_but_not_referenced" ${CUDA_INCLUDE}
GPU2: CXX_LINK = ${CUDA_LIB}
GPU2: CXX = nvcc
GPU2: all

all: ${OBJ} ${OBJ_TRANSPOSE}
	${CXX} ${OBJ} ${OBJ_TRANSPOSE} ${CXX_FLAGS} -o gett.exe ${CXX_LINK}

%.o: %.cpp
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@ 
%.o: ttc_transpositions/%.cu
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@ 

clean:
	rm -f *.o ${OBJ} ${OBJ_TRANSPOSE} gett.exe
