//
// Program to perform Backward Euler time-marching on a 1D grid
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
  __shared__  float a[128], c[128], d[128];

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

  int    NX = 32, niter = 2;

  float *h_u, *h_v, *h_c, *d_u;

  // initialise card

  findCudaDevice(argc, argv);

  // allocate memory on host and device

  h_u = (float *)malloc(sizeof(float)*NX);
  h_v = (float *)malloc(sizeof(float)*NX);
  h_c = (float *)malloc(sizeof(float)*NX);

  checkCudaErrors( cudaMalloc((void **)&d_u, sizeof(float)*NX) );

  // GPU execution

  for (int i=0; i<NX; i++) h_u[i] = 1.0f;

  checkCudaErrors( cudaMemcpy(d_u, h_u, sizeof(float)*NX,
                              cudaMemcpyHostToDevice) );

  GPU_trid<<<1, NX>>>(NX, niter, d_u);

  checkCudaErrors( cudaMemcpy(h_u, d_u, sizeof(float)*NX,
                              cudaMemcpyDeviceToHost) );


  // CPU execution

  for (int i=0; i<NX; i++) h_v[i] = 1.0f;

  gold_trid(NX, niter, h_v, h_c, 0);


  // print out array

  for (int i=0; i<NX; i++) {
    printf(" %d  %f  %f  %f \n",i,h_u[i],h_v[i], h_u[i]-h_v[i]);
  }

 // Release GPU and CPU memory

  checkCudaErrors( cudaFree(d_u) );

  free(h_u);
  free(h_v);
  free(h_c);

}
