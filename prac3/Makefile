
INC	:= -I$(CUDA_HOME)/include -I. -I../headers
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

all: 		laplace3d laplace3d_new

laplace3d:	laplace3d.cu laplace3d_gold.cpp Makefile
		nvcc laplace3d.cu laplace3d_gold.cpp -o laplace3d \
		$(INC) $(NVCCFLAGS) $(LIB)

laplace3d_new:	laplace3d_new.cu laplace3d_gold.cpp Makefile
		nvcc laplace3d_new.cu laplace3d_gold.cpp -o laplace3d_new \
		$(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f laplace3d laplace3d_new
