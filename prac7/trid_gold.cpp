// NOTE: modified to work with multiple starting values 

/* Using the Thomas algorithm to solve a tridiagonal system
/  iteratively solve A * u_n+1 = lambda * u_n 
/
/ PARAMETERS:
/ u -> at start of iteration: stores u_n (vector from previous time step)
/   -> at end of iteration: stores u_n+1 (newly computer solution vector)
/ 
/ c -> used to store modified super-diagonal coefficients calculated during the 'forward pass'
/
/
/ LOCAL VARIABLES: 
/ aa -> sub-diagonal element of matrix A
/ bb -> main diagonal element of matrix A
/ cc -> super-diagonal element of matrix A
/ dd -> i-th element of RHS of equation 
*/

void gold_trid(int NX, int niter, float* u, float* c, int start)
{
  // modify starting values
  u = &u[start*NX];
  c = &c[start*NX];

  float lambda=1.0f, aa, bb, cc, dd;

  for (int iter=0; iter<niter; iter++) {

    //
    // forward pass - get rid of subdiagonal terms
    //

    aa   = -1.0f;
    bb   =  2.0f + lambda;
    cc   = -1.0f;
    dd   = lambda*u[0]; // first row has no subdiagonal term

    // normalise with respect to main diagonal value
    bb   = 1.0f / bb;
    cc   = bb*cc;
    dd   = bb*dd;
    c[0] = cc;
    u[0] = dd;

    for (int i=1; i<NX; i++) {
      aa   = -1.0f;
      bb   = 2.0f + lambda - aa*cc;
      dd   = lambda*u[i] - aa*dd;
      bb   = 1.0f/bb;
      cc   = -bb;
      dd   = bb*dd;
      c[i] = cc;
      u[i] = dd;
    }

    //
    // reverse pass - modified backward sub
    //

    u[NX-1] = dd;

    for (int i=NX-2; i>=0; i--) {
      dd   = u[i] - c[i]*dd;
      u[i] = dd;
    }
  }
}


