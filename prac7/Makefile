
INC	:= -I$(CUDA_HOME)/include -I. -I../headers
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

all: trid trid_dynamic trid_shuffle

trid:	trid.cu trid_gold.cpp Makefile
	nvcc trid.cu trid_gold.cpp -o trid \
	$(INC) $(NVCCFLAGS) $(LIB)

trid_dynamic:	trid_dynamic.cu trid_gold.cpp Makefile
	nvcc trid_dynamic.cu trid_gold.cpp -o trid_dynamic \
	$(INC) $(NVCCFLAGS) $(LIB)

trid_shuffle:	trid_shuffle.cu trid_gold.cpp Makefile
	nvcc trid_shuffle.cu trid_gold.cpp -o trid_shuffle \
	$(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm trid trid_dynamic trid_shuffle
