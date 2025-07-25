
INC	:= -I$(CUDA_HOME)/include -I. -I../headers
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

all:	prac2 prac2_device

prac2:	prac2.cu Makefile
	nvcc prac2.cu -o prac2 $(INC) $(NVCCFLAGS) $(LIB)

prac2_device:	prac2_device.cu Makefile
	nvcc prac2_device.cu -o prac2_device $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac2 prac2_device
