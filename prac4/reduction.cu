////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- initial code for shared memory reduction for 
//                a single block which is a power of two in size
//         MODIFIED: correct reduction for arbitrary number of blocks and threads 
//                    and using atomic addition variant
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

// __device__ int lock = 0;

__global__ void reduction(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[tid+blockIdx.x*blockDim.x];

    // Round blockDim.x/2 up to the nearest power of 2
    int m;
    for (m=1; m<blockDim.x; m=2*m) {}
    m = m/2;

    // next, we perform binary tree reduction

    for (int d=m; d>0; d=d/2) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d && (tid+d)<blockDim.x)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory

    // save block reduction and then do the block-level reduction in the host
    // if (tid==0) g_odata[blockIdx.x] = temp[0];

    // Alternatively: block level reduction using atomic adds
    if (tid==0) atomicAdd(g_odata, temp[0]);
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_blocks, num_threads, num_elements, mem_size, shared_mem_size;

  float *h_data, *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  // num_blocks   = 1;  // start with only 1 thread block
  // Modify number of blocks
  num_blocks   = 10;  // start with only 1 thread block
  num_threads  = 3;
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
      
  for(int i = 0; i < num_elements; i++) {
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));
  }

  // compute reference solution

  float sum = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, sizeof(float) * num_blocks) );

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );
                          
  checkCudaErrors( cudaMemcpy(d_odata, d_odata, sizeof(float) * num_blocks,
                              cudaMemcpyHostToDevice) );

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;

  cudaEventRecord(start);

  reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_odata,d_idata);
  getLastCudaError("reduction kernel execution failed");

  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&milli, start, stop);
  // printf("Reduction kernel execution time: %.1f (ms) \n\n", milli);

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, num_blocks * sizeof(float),
                              cudaMemcpyDeviceToHost) );

  // do the final summation back in the sum
  // float final_sum = 0.0f;
  // for(int i=0; i<num_blocks; i++)  {
  //   final_sum += h_data[i];
  // }
  // h_data[0] = final_sum;

  // using atomic addition
  h_data[0] = h_data[0];

  // check results

  printf("\nreduction error = %f\n",h_data[0]-sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
