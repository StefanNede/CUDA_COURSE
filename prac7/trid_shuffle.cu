//
// Program to perform Backward Euler time-marching on a 1D grid
// 
// Modified from trid_dynamic to be limited to the case of NX=32
// so we can use a single warp and use shuffles instead of shared memory
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

    __syncthreads();  // finish writes 

    // parallel cyclic reduction
    float a_val, b_val, c_val, d_val;
    float orig_aa, orig_cc, orig_dd;
    

    for (int nt=1; nt<NX; nt=2*nt) {
      orig_aa = aa;
      orig_cc = cc;
      orig_dd = dd;

      bb = 1.0f;

      // have to get original values as overwrite them in the if blocks 
      d_val = __shfl_sync(-1, orig_dd, tid-nt);
      c_val = __shfl_sync(-1, orig_cc, tid-nt);
      a_val = __shfl_sync(-1, orig_aa, tid-nt);
      if (tid - nt >= 0) {
        dd = dd - orig_aa * d_val;
        bb = bb - orig_aa * c_val;
        aa = -orig_aa * a_val;
      }

      d_val = __shfl_sync(-1, orig_dd, tid+nt);
      a_val = __shfl_sync(-1, orig_aa, tid+nt);
      c_val = __shfl_sync(-1, orig_cc, tid+nt);
      if (tid + nt < NX) {
        dd = dd - orig_cc * d_val;
        bb = bb - orig_cc * a_val;
        cc = -orig_cc * c_val;
      }

      __syncthreads();

      bbi = 1.0f / bb;
      aa  = aa*bbi;
      cc  = cc*bbi;
      dd  = dd*bbi;
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

  int    NX = 32, niter = 2, M = 1;

  float *h_u, *h_v, *h_c, *d_u, shmem;

  // initialise card

  findCudaDevice(argc, argv);

  // allocate memory on host and device

  h_u = (float *)malloc(sizeof(float)*NX*M);
  h_v = (float *)malloc(sizeof(float)*NX*M);
  h_c = (float *)malloc(sizeof(float)*NX*M);

  checkCudaErrors( cudaMalloc((void **)&d_u, sizeof(float)*NX*M) );

  // GPU execution

  for (int j=0; j<M; j++) {
    for (int i=0; i<NX; i++) {
      int k = i + j*NX;
      h_u[k] = (float) (j+1);
    }
  }

  checkCudaErrors( cudaMemcpy(d_u, h_u, sizeof(float)*NX*M,
                              cudaMemcpyHostToDevice) );

  GPU_trid<<<M, NX>>>(NX, niter, d_u);

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
