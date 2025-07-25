
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>


///////////////////////////////////////////////////////////////////////
// CPU routine
///////////////////////////////////////////////////////////////////////

void scan_gold(float* odata, float* idata, const unsigned int len) 
{
  odata[0] = 0;
  for(int i=1; i<len; i++) odata[i] = idata[i-1] + odata[i-1];
}

///////////////////////////////////////////////////////////////////////
// GPU routine
///////////////////////////////////////////////////////////////////////

__global__ void scan(float *g_odata, float *g_idata)
{
  // Dynamically allocated shared memory for scan kernels

  extern __shared__  float tmp[];

  float temp;
  int   tid = threadIdx.x;

  // read input into shared memory

  temp     = g_idata[tid];
  tmp[tid] = temp;

  // perform scan

  for (int d=1; d<blockDim.x; d=2*d) {
    __syncthreads();
    if (tid-d >= 0) temp += tmp[tid-d];
    __syncthreads();
    tmp[tid] = temp;
  }

  // write results to global memory

  __syncthreads();

  temp = 0.0f;
  if (tid>0) temp = tmp[tid-1];

  g_odata[tid] = temp;
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_threads, num_elements, mem_size, shared_mem_size;

  float *h_data, *reference;
  float *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_threads  = 512;
  num_elements = num_threads;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float*) malloc(mem_size);
      
  for(int i=0; i<num_elements; i++) 
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));

  // compute reference solution

  reference = (float*) malloc(mem_size);
  scan_gold(reference, h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, mem_size) );

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice));

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  scan<<<1,num_threads,shared_mem_size>>>(d_odata,d_idata);
  getLastCudaError("scan kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, mem_size,
                              cudaMemcpyDeviceToHost) );

  // check results

  float err=0.0;
  for (int i=0; i<num_elements; i++)
    err += (h_data[i] - reference[i])*(h_data[i] - reference[i]);
  printf("rms scan error  = %f\n",sqrt(err/num_elements));

  // cleanup memory

  free(h_data);
  free(reference);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
