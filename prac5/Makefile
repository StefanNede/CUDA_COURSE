
INC	:= -I$(CUDA_HOME)/include -I. -I../headers
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcurand -lcufft

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

both:		tensorCUBLAS simpleTensorCoreGEMM

tensorCUBLAS:	tensorCUBLAS.cu Makefile
		nvcc tensorCUBLAS.cu -o tensorCUBLAS $(INC) $(NVCCFLAGS) $(LIB)

simpleTensorCoreGEMM:	simpleTensorCoreGEMM.cu Makefile
		nvcc simpleTensorCoreGEMM.cu -o simpleTensorCoreGEMM $(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f tensorCUBLAS simpleTensorCoreGEMM simpleCUFFT
