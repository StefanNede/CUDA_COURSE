//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void oned_add_vector(float *x, float *y, int size)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  //printf("THREAD ID: %d, NUMBER 1: %f, NUMBER 2: %f\n", tid, x[tid], y[tid]);

  if (tid < size) 
  {
    x[tid] = (float) (x[tid] + y[tid]);
  }
}


//
// main code
//

int main(int argc, const char **argv)
{
  float *h_x1, *h_x2, *d_x1, *d_x2;
  int   nblocks, nthreads, nsize, n, vector_size; 
  // vectors are vector_sizex1 (1D vectors only at the moment)

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks = 2;
  nthreads = 8;
  vector_size = 3;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_x1 = (float *)malloc(vector_size*sizeof(float));
  h_x2 = (float *)malloc(vector_size*sizeof(float));

  for (int i=0; i<vector_size; i++)
  {
    h_x1[i] = (float) i;
    h_x2[i] = (float) (nsize-i);
  }

  // printing 1D vectors we are adding
  printf("VECTOR 1: ");
  for (n=0; n<vector_size; n++) printf("%f, ",h_x1[n]);
  printf("\n");
  printf("VECTOR 2: ");
  for (n=0; n<vector_size; n++) printf("%f, ",h_x2[n]);
  printf("\n");

  checkCudaErrors(cudaMalloc((void **)&d_x1, vector_size*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_x2, vector_size*sizeof(float)));

  checkCudaErrors( cudaMemcpy(d_x1,h_x1,vector_size*sizeof(float),
                 cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(d_x2,h_x2,vector_size*sizeof(float),
                 cudaMemcpyHostToDevice) );

  // execute kernel
  
  oned_add_vector<<<nblocks,nthreads>>>(d_x1, d_x2, vector_size);
  getLastCudaError("add_vector execution failed\n");

  // copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x1,d_x1,vector_size*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  //for (n=0; n<vector_size; n++) printf(" n,  x  =  %d  %f \n",n,h_x1[n]);

  // printing 1D result vector post adding
  printf("RESULT VECTOR: ");
  for (n=0; n<vector_size; n++) printf("%f, ",h_x1[n]);
  printf("\n");

  // free memory 

  checkCudaErrors(cudaFree(d_x1));
  checkCudaErrors(cudaFree(d_x2));
  free(h_x1);
  free(h_x2);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
