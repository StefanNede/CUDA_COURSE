//
// Program to perform Backward Euler time-marching on a 1D grid
// 
// Modified to perform M independent calculations (each with their own starting value for x)
// using M thread blocks -> make array u have multiple sequential starting values
// and calculate correct offset index 
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// kernel function
////////////////////////////////////////////////////////////////////////

__global__ void GPU_trid(int NX, int niter, float *u)
{
  // using dynamic shared memory
  extern __shared__ float shmem[];
  float* a = shmem;
  float* c = &a[NX];
  float* d = &c[NX];

  // select which starting value of x to work with given the block index and dimension
  // i.e. we are finding the offset for the vector we want to work with
  int id = blockIdx.x*blockDim.x;
  u = &u[id];

  float aa, bb, cc, dd, bbi, lambda=1.0;
  int   tid;

  for (int iter=0; iter<niter; iter++) {

    // set tridiagonal coefficients and r.h.s.

    tid = threadIdx.x;
    bbi = 1.0f / (2.0f + lambda); // normalised
    
    if (tid>0)
      aa = -bbi;
    else // otherwise no sub-diagonal value
      aa = 0.0f; 

    if (tid<blockDim.x-1)
      cc = -bbi;
    else // otherwise no super-diagonal value
      cc = 0.0f;

    if (iter==0) 
      dd = lambda*u[tid]*bbi;
    else
      dd = lambda*dd*bbi; // build on previous timestep vallue

    a[tid] = aa;
    c[tid] = cc;
    d[tid] = dd;

    // parallel cyclic reduction

    for (int nt=1; nt<NX; nt=2*nt) {
      __syncthreads();  // finish writes before reads

      bb = 1.0f;

      if (tid-nt >= 0) {
        dd = dd - aa*d[tid-nt];
        bb = bb - aa*c[tid-nt];
        aa =    - aa*a[tid-nt];
      }

      if (tid+nt < NX) {
        dd = dd - cc*d[tid+nt];
        bb = bb - cc*a[tid+nt];
        cc =    - cc*c[tid+nt];
      }

      __syncthreads();  // finish reads before writes


      bbi = 1.0f / bb;
      aa  = aa*bbi;
      cc  = cc*bbi;
      dd  = dd*bbi;

      a[tid] = aa;
      c[tid] = cc;
      d[tid] = dd;
    }
  }

  u[tid] = dd;
}

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void gold_trid(int, int, float*, float*, int);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  int    NX = 100, niter = 2, M = 6;

  float *h_u, *h_v, *h_c, *d_u, shmem;

  // initialise card

  findCudaDevice(argc, argv);

  // allocate memory on host and device

  h_u = (float *)malloc(sizeof(float)*NX*M);
  h_v = (float *)malloc(sizeof(float)*NX*M);
  h_c = (float *)malloc(sizeof(float)*NX*M);

  checkCudaErrors( cudaMalloc((void **)&d_u, sizeof(float)*NX*M) );

  // GPU execution
  shmem = NX*3*sizeof(float);

  for (int j=0; j<M; j++) {
    for (int i=0; i<NX; i++) {
      int k = i + j*NX;
      h_u[k] = (float) (j+1);
    }
  }

  checkCudaErrors( cudaMemcpy(d_u, h_u, sizeof(float)*NX*M,
                              cudaMemcpyHostToDevice) );

  GPU_trid<<<M, NX, shmem>>>(NX, niter, d_u);

  checkCudaErrors( cudaMemcpy(h_u, d_u, sizeof(float)*NX*M,
                              cudaMemcpyDeviceToHost) );


  // CPU execution

  for (int j=0; j<M; j++) {
    for (int i=0; i<NX; i++) {
      int k = i + j*NX;
      h_v[k] = (float) (j+1);
    }
    gold_trid(NX, niter, h_v, h_c, j);
  }

  // print out array

  for (int j=0; j<M; j++) {
    printf("With starting value index %d\n", j);
    for (int i=0; i<NX; i++) {
      int k = i + j*NX;
      printf(" %d  %f  %f  %f \n",i,h_u[k],h_v[k], h_u[k]-h_v[k]);
    }
  }

 // Release GPU and CPU memory

  checkCudaErrors( cudaFree(d_u) );

  free(h_u);
  free(h_v);
  free(h_c);

}
