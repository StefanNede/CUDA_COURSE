////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2;


////////////////////////////////////////////////////////////////////////
// kernel routines -- see sections 3.5, 3.6 in cuRAND documentation
////////////////////////////////////////////////////////////////////////

__global__ void RNG_init(curandState *state)
{
  // RNG initialisation with id-based skipahead
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(1234, id, 0, &state[id]);
}


__global__ void pathcalc(curandState *device_state, float *d_v,
                         int mpath, int NPATH)
{
  float s1, s2, y1, y2, payoff;

  int id = threadIdx.x + blockIdx.x*blockDim.x;
  curandState_t state = device_state[id];

  for(int m=0; m<mpath; m++) {
    s1 = 1.0f;
    s2 = 1.0f;

    for (int n=0; n<N; n++) {
      y1 = curand_normal(&state);
      y2 = rho*y1 + alpha*curand_normal(&state);

      s1 = s1*(con1 + con2*y1);
      s2 = s2*(con1 + con2*y2);
    }

    // put payoff value into device array

    payoff = 0.0f;
    if ( fabs(s1-1.0f)<0.1f && fabs(s2-1.0f)<0.1f ) payoff = exp(-r*T);

    int payoff_id = id + m*gridDim.x*blockDim.x;
    if (payoff_id < NPATH) d_v[payoff_id] = payoff;
  }
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
    
  int     NPATH=9600000, h_N=100;
  float   h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2;
  float  *h_v, *d_v;
  double  sum1, sum2;
  curandState *state;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float)*NPATH);
  checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&state, sizeof(curandState)*NPATH) );

  printf("size of curandState is %d bytes\n",sizeof(curandState));

  // define constants and transfer to GPU

  h_T     = 1.0f;
  h_r     = 0.05f;
  h_sigma = 0.1f;
  h_rho   = 0.5f;
  h_alpha = sqrt(1.0f-h_rho*h_rho);
  h_dt    = 1.0f/h_N;
  h_con1  = 1.0f + h_r*h_dt;
  h_con2  = sqrt(h_dt)*h_sigma;

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(rho,  &h_rho,  sizeof(h_rho)) );
  checkCudaErrors( cudaMemcpyToSymbol(alpha,&h_alpha,sizeof(h_alpha)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );
  checkCudaErrors( cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)) );
  checkCudaErrors( cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)) );

  // calculate theoretical occupancy -- see Pro Tip blog article:
  // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
  
  int maxActiveBlocks, blockSize=128;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 pathcalc, blockSize, 0);
  printf("maxActiveBlocks/SM = %d \n",maxActiveBlocks);
  printf("number of SMs      = %d \n",props.multiProcessorCount);
  int blocks = maxActiveBlocks*props.multiProcessorCount;
  
  // execute kernels

  cudaEventRecord(start);
  RNG_init<<<blocks, 128>>>(state);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  getLastCudaError("RNG_init execution failed\n");
  printf("RNG_init kernel execution time (ms): %f \n",milli);

  int paths_per_thread = (NPATH-1)/(128*blocks) + 1;
  cudaEventRecord(start);
  pathcalc<<<blocks, 128>>>(state,d_v,paths_per_thread,NPATH);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  getLastCudaError("pathcalc execution failed\n");
  printf("pathcalc kernel execution time (ms): %f \n",milli);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
                   cudaMemcpyDeviceToHost) );

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i=0; i<NPATH; i++) {
    sum1 += h_v[i];
    sum2 += h_v[i]*h_v[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	 sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );

  // Release memory and exit cleanly

  free(h_v);
  checkCudaErrors( cudaFree(d_v) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  // performance output

  printf("RNGs/sec = %g \n",2.0*h_N*NPATH/(milli*0.001));


}
