
INC	:= -I$(CUDA_HOME)/include -I./headers
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math \
		   -D BLOCK_X=$(BLOCK_X) -D BLOCK_Y=$(BLOCK_Y)

laplace3d_$(ID):	laplace3d.cu
		nvcc laplace3d.cu -o bin/laplace3d_$(ID) \
		$(INC) $(NVCCFLAGS) $(LIB)
