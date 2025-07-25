
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2, a, b, c;


////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

__global__ void find_average(float *d_z, float *d_v) 
{
  float y1, value, sum, average;
  int   ind;

  // move array pointers to correct position

  ind = threadIdx.x + N*blockIdx.x*blockDim.x;
  sum = 0.0f;

  // average with random number calculation

  for (int n=0; n<N; n++) {
    y1   = d_z[ind];
    // printf("Index: %d\n", ind);
    ind += blockDim.x;      // shift pointer to next element
    value = a*y1*y1 + b*y1 + c;
    sum += value;
  }

  average = sum/N;
  // printf("AVERAGE: %f\n", average);

  d_v[threadIdx.x + blockIdx.x*blockDim.x] = average;
}

__global__ void pathcalc(float *d_z, float *d_v)
{
  float s1, s2, y1, y2, payoff;
  int   ind;

  // move array pointers to correct position

  // version 1
  ind = threadIdx.x + 2*N*blockIdx.x*blockDim.x;

  // version 2
  // ind = 2*N*threadIdx.x + 2*N*blockIdx.x*blockDim.x;


  // path calculation

  s1 = 1.0f;
  s2 = 1.0f;

  for (int n=0; n<N; n++) {
    y1   = d_z[ind];
    // printf("Index: %d\n", ind);
    // version 1
    ind += blockDim.x;      // shift pointer to next element
    // version 2
    // ind += 1; 

    y2   = rho*y1 + alpha*d_z[ind];
    // version 1
    ind += blockDim.x;      // shift pointer to next element
    // version 2
    // ind += 1; 

    s1 = s1*(con1 + con2*y1);
    s2 = s2*(con1 + con2*y2);
  }

  // put payoff value into device array

  payoff = 0.0f;
  if ( fabs(s1-1.0f)<0.1f && fabs(s2-1.0f)<0.1f ) payoff = exp(-r*T);

  d_v[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
  // printf("VERSION 1\n");
    
  // NPATH = number of path simulations we are doing
  // N = number of timesteps 
  int     NPATH=9600000, h_N=200;
  float   h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2, h_a, h_b, h_c;
  float  *h_v, *d_v, *d_z;
  double  sum1, sum2;

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
  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*h_N*NPATH) );

  // define constants and transfer to GPU

  h_T     = 1.0f;
  h_r     = 0.05f;
  h_sigma = 0.1f;
  h_rho   = 0.5f;
  h_alpha = sqrt(1.0f-h_rho*h_rho);
  h_dt    = 1.0f/h_N;
  h_con1  = 1.0f + h_r*h_dt;
  h_con2  = sqrt(h_dt)*h_sigma;
  h_a = 1.0f;
  h_b = 2.0f;
  h_c = 3.0f;

  // potentially could pass these in as a struct but not recommended on forums
  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(rho,  &h_rho,  sizeof(h_rho)) );
  checkCudaErrors( cudaMemcpyToSymbol(alpha,&h_alpha,sizeof(h_alpha)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );
  checkCudaErrors( cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)) );
  checkCudaErrors( cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)) );
  checkCudaErrors( cudaMemcpyToSymbol(a, &h_a, sizeof(h_a)) );
  checkCudaErrors( cudaMemcpyToSymbol(b, &h_b, sizeof(h_b)) );
  checkCudaErrors( cudaMemcpyToSymbol(c, &h_c, sizeof(h_c)) );

  // random number generation

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

  cudaEventRecord(start); // for stopwatch
  checkCudaErrors( curandGenerateNormal(gen, d_z, h_N*NPATH, 0.0f, 1.0f) );
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop); // get ellapsed time in milliseconds

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, h_N*NPATH/(0.001*milli));

  // execute kernel and time it -- Path simulations

  // cudaEventRecord(start);
  // 128 threads per block so want NPATH/128 blocks as each thread is doing 1 and only 1 path
  // pathcalc<<<NPATH/128, 128>>>(d_z, d_v);
  // cudaEventRecord(stop);

  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&milli, start, stop);

  // getLastCudaError("pathcalc execution failed\n");
  // printf("Monte Carlo kernel execution time (ms): %f \n",milli);

  // execute kernel and time it -- Finding mean of normal random variables
  
  cudaEventRecord(start);
  find_average<<<NPATH/128, 128>>>(d_z, d_v);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  getLastCudaError("find_average execution failed\n");
  printf("Monte Carlo kernel execution time (ms): %f \n",milli);


  // copy back results

  checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
                   cudaMemcpyDeviceToHost) );

  // compute average + variance/std. dev

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i=0; i<NPATH; i++) {
    // printf("%f\n", h_v[i]);
    sum1 += h_v[i];
    sum2 += h_v[i]*h_v[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	 sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_v);
  checkCudaErrors( cudaFree(d_v) );
  checkCudaErrors( cudaFree(d_z) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}
