#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Function to check for errors.
void checkError(cublasStatus_t status){
    if (status != CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS error\n");
        exit(EXIT_FAILURE);
    }
}

int main(void){

    // Variable to hold the status returned by cuBLAS.
    cublasStatus_t status;

    // Variable that serves as a handle to the cuBLAS lib context.
    // This describes the environment in which our cuBLAS routines
    // are executed. It is initialised with cublasCreate().
    cublasHandle_t handle;

    // Pointers to type float for input and output matrices.
    float *d_A, *d_B, *d_C;

    // Size of row or col in matrix (we will work with a square matrix).
    int n = 128;

    // To get an accurate timing measurement we will run the same code
    // multiple times. Five times should be more than enough.
    // changed to 3 for testing + plotting
    int num_reps = 3;

    // We will execute the sgemm operation.
    // C = alpha A * B + beta C
    // alpha, beta are scalar, A, B, C are matrices.
    float alpha = 1.0f;
    float beta = 0.0f;

    // Declare a variable to hold out timings.
    float milliseconds = 0.0f;

    // Declare two cuda events for timing.
    cudaEvent_t start, stop;


    // Allocate memory for the input and output matrices on the host.
    float *h_A = (float*)malloc(n * n * sizeof(float));
    float *h_B = (float*)malloc(n * n * sizeof(float));
    float *h_C = (float*)malloc(n * n * sizeof(float));

    // Initialise the matrices with random data.
    for (int i = 0; i < n * n; i++) {
        float A = (float)(rand() % 100);
        float B = (float)(rand() % 100);
        h_A[i] = A;
        h_B[i] = B;
    }

    // Allocate memory on the device for input and output matrices.
    cudaMalloc((void**)&d_A, n * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * n * sizeof(float));
    cudaMalloc((void**)&d_C, n * n * sizeof(float));

    // Copy the input data to the device
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create a cuBLAS handle
    status = cublasCreate(&handle);
    checkError(status);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Perform the matrix multiplication with Tensor Cores disabled.
    // Here we are using CUBLAS_PEDANTIC_MATH to disable the tensor cores on Volta.
    status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    checkError(status);

    // Run the same code num_reps times to get a representative timing.
    // Please read  https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    // to understand the input parameters to cublasSgemm().
    // CUBLAS_OP_N = non-transpose operation
    for (int i = 0; i < num_reps; i++) {
        cudaEventRecord(start);
        checkError(status);
        // d_C = alpha * (d_A x d_B) + beta * d_C
        // to 'prevent' overwriting of matrix C copy its value to another matrix d_D and then pass d_D as the argument into the gem
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time for SGEMM without Tensor Cores: %f ms\n", milliseconds);
    }

    // Copy the result back to the host.
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Check the result.
    // the pragma stuff runs the for loop over multiple cores (each core does a different value of i independently) - C syntax
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            float expected = 0.0f;
            #pragma omp parallel for reduction(+:expected)
            for (int k = 0; k < n; k++) {
                expected += h_A[k*n + j] * h_B[i*n + k];
            }
            // Check if the absolute difference of the cuBLAS output is within a small tolerance of the expected output.
            if (fabs(h_C[i*n + j] - expected) > 1e-1) {
               printf("Verification failed at index %d,%d! h_C[%d,%d] = %f, expected = %f\n", i, j, i, j, h_C[i*n + j], expected);
               exit(-1);
            }
        }
    }
    printf("Verification passed!\n\n");

    /****************************************************************************************************************************
                This next section is exactly the same as the last, however this time we pass cublasSetMathMode()
                CUBLAS_TF32_TENSOR_OP_MATH. This is the easiest way to enable acceleration of single-precision
                routines using TF32 tensor cores for those working on architectures Ampere and above.

                                    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

                                             NOTE: THIS WILL HAVE NO EFFECT ON VOLTA
     ***************************************************************************************************************************/

    // status = cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    // checkError(status);
    // for (int i = 0; i < num_reps; i++) {
    //     cudaEventRecord(start);
    //     checkError(status);
    //     status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     cudaEventElapsedTime(&milliseconds, start, stop);
    //     printf("Timie for SGEMM with Tensor Cores and TF32 (Ampere and beyond): %f ms\n", milliseconds);
    // }

    // cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // #pragma omp parallel for
    // for (int i = 0; i < n; i++) {
    //     #pragma omp parallel for
    //     for (int j = 0; j < n; j++) {
    //         float expected = 0.0f;
    //         #pragma omp parallel for reduction(+:expected)
    //         for (int k = 0; k < n; k++) {
    //             expected += h_A[k*n + j] * h_B[i*n + k];
    //         }
    //         if (fabs(h_C[i*n + j] - expected) > 1e-1) {
    //            printf("Verification failed at index %d,%d! h_C[%d,%d] = %f, expected = %f\n", i, j, i, j, h_C[i*n + j], expected);
    //            exit(-1);
    //         }
    //     }
    // }
    // printf("Verification passed!\n\n");


    /****************************************************************************************************************************
                This next section uses both tensor cores and mixed precision for acceleration on the volta architecture.
                To do this we tell cublasSetMathMode() to use the default math mode by passing CUBLAS_DEFAULT_MATH
                We then use the cublasSgemmEx() function to perform the sgemm. Note we explicitly tell it that our
                inputs are fp16.

                See here for further information:

                https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmex

     ***************************************************************************************************************************/

    // Allocate memory for fp16 arrays on the host
    half *h_hA = (half*)malloc(n * n * sizeof(half));
    half *h_hB = (half*)malloc(n * n * sizeof(half));

    // Allocate memory for fp16 on the device
    half *d_hA, *d_hB;

    cudaMalloc((void**)&d_hA, n * n * sizeof(half));
    cudaMalloc((void**)&d_hB, n * n * sizeof(half));

    // Initialise the matrices with data
    // that is suitable for the fp16 range
    for (int i = 0; i < n * n; i++) {
        float A = (float)(rand() % 100)/100.0f;
        float B = (float)(rand() % 100)/100.0f;
        h_hA[i] = (half)(A);
        h_hB[i] = (half)(B);
    }

    // Copy the data to the device
    cudaMemcpy(d_hA, h_hA, n * n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hB, h_hB, n * n * sizeof(half), cudaMemcpyHostToDevice);

    // Perform the matrix multiplication with Tensor Cores enabled and mixed precision
    status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    checkError(status);
    for (int i = 0; i < num_reps; i++) {
        cudaEventRecord(start);
        status = cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_hA, CUDA_R_16F, n, d_hB, CUDA_R_16F, n, &beta, d_C, CUDA_R_32F, n);
        checkError(status);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time for SGEMM with Tensor Cores and mixed precision (Volta): %f ms\n", milliseconds);
    }

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Check the result
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            float expected = 0.0f;
            #pragma omp parallel for reduction(+:expected)
            for (int k = 0; k < n; k++) {
                expected += (float)h_hA[k*n + j] * (float)h_hB[i*n + k];
            }
            //printf("\n%f\t%f", expected, h_C[i*n + j]);
            // Check if the absolute difference is within a small tolerance
            if (fabs(h_C[i*n + j] - expected) > 1e-1) {
               printf("Verification failed at index %d,%d! h_C[%d,%d] = %f, expected = %f\n", i, j, i, j, h_C[i*n + j], expected);
               exit(-1);
            }
        }
    }
    printf("Verification passed!\n\n");


    // Clean up
    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_hA);
    cudaFree(d_hB);
    free(h_hA);
    free(h_hB);

    return 0;
}

