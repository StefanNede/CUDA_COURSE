
INC	:= -I$(CUDA_HOME)/include -I. -I../headers
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

all: 		reduction reduction_warps

reduction:	reduction.cu Makefile
		nvcc reduction.cu -o reduction $(INC) $(NVCCFLAGS) $(LIB)

reduction_warps: reduction_warps.cu Makefile
		nvcc reduction_warps.cu -o reduction_warps $(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f reduction reduction_warps

