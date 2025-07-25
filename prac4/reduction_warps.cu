////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- reduction using shuffle instructions for warp level reduction
//                 and atomics for block-level reduction
//
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CPU routine
////////////////////////////////////////////////////////////////////////

float reduction_gold(float* idata, int len) 
{
  float sum = 0.0f;
  for(int i=0; i<len; i++) sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////
// GPU routine
////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *running_sum, float *g_idata)
{
    int tid = threadIdx.x;

    // get sum of current warp
    float warp_sum = g_idata[tid + blockIdx.x*blockDim.x]; // value of thread

    // sum in increasingly large groups -> first groups of 2, then 4 etc.
    for (int d = 1; d < warpSize; d = 2*d) {
      __syncthreads();
      warp_sum += __shfl_xor_sync(-1, warp_sum, d);
    }

    // block level reduction -- use modulo 32 if there are more threads than 32 => multiple warps per block
    if (tid%32==0) atomicAdd(running_sum, warp_sum);
}

// doing the whole reduction for the whole block in one kernel function using 2 shuffles
__global__ void reduction2(float *running_sum, float *g_idata)
{
    extern __shared__ float temp[32];
    int tid = threadIdx.x;

    // get sum of current warp
    float warp_sum = g_idata[tid + blockIdx.x*blockDim.x]; // value of thread

    // sum in increasingly large groups -> first groups of 2, then 4 etc.
    for (int d = 1; d < warpSize; d = 2*d) {
      warp_sum += __shfl_xor_sync(-1, warp_sum, d);
    }

    // move the warp sums of each warp into shared memory
    if (tid%32 == 0) temp[tid/32] = warp_sum;

    __syncthreads();

    // move the warp sums in shared memory into the first warp values
    if (tid < 32) {
      warp_sum = temp[tid];
    }

    __syncthreads();

    // sum the warp sums from shared memory
    for (int d = 1; d < warpSize; d = 2*d) {
      warp_sum += __shfl_xor_sync(-1, warp_sum, d);
    }

    // block level reduction -- use modulo 32 if there are more threads than 32 => multiple warps per block
    if (tid == 0) atomicAdd(running_sum, warp_sum);
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_blocks, num_threads, num_elements, mem_size, shared_mem_size;

  float *h_data, *d_idata, *d_odata, *d_running_sum, *h_running_sum;

  // initialise card

  findCudaDevice(argc, argv);

  // num_blocks   = 1;  // start with only 1 thread block
  // Modify number of blocks
  num_blocks   = 1024;  // start with only 1 thread block
  num_threads  = 1024;
  num_elements = num_blocks*num_threads;
  mem_size     = sizeof(float) * num_elements;

  // initialise CUDA timing
  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float*) malloc(mem_size);

  // initialise sum to float value 0
  h_running_sum = (float*) malloc(sizeof(float));
  *h_running_sum = 0.0f; // dereference pointer here to set value
      
  for(int i = 0; i < num_elements; i++) {
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));
  }

  // compute reference solution

  float sum = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_running_sum, sizeof(float)));

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );
                          
  checkCudaErrors( cudaMemcpy(d_running_sum, h_running_sum, sizeof(float), cudaMemcpyHostToDevice));

  // execute the kernel

  //reduction<<<num_blocks,num_threads>>>(d_running_sum,d_idata);
  reduction2<<<num_blocks,num_threads>>>(d_running_sum,d_idata);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_running_sum, d_running_sum, sizeof(float), cudaMemcpyDeviceToHost) );

  // check results

  printf("\nreduction error = %f\n",*h_running_sum-sum);

  // cleanup memory

  free(h_data);
  free(h_running_sum);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_running_sum) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
