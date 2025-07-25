
////////////////////////////////////////////////////////////////////////////////
void Gold_laplace3d(int NX, int NY, int NZ, float* u1, float* u2) 
{
  float sixth = 1.0f/6.0f;  // predefining this improves performance more than 10%

  for (int k=0; k<NZ; k++) {
    for (int j=0; j<NY; j++) {
      for (int i=0; i<NX; i++) {   // i loop innermost for sequential memory access
	long long ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1) {
          u2[ind] = u1[ind];          // Dirichlet b.c.'s
        }
        else {
          // imagine a cube, it has 6 faces -> average those bad boys
          u2[ind] = ( u1[ind-1    ] + u1[ind+1    ]
                    + u1[ind-NX   ] + u1[ind+NX   ]
                    + u1[ind-NX*NY] + u1[ind+NX*NY] ) * sixth; 
        }
      }
    }
  }
}

