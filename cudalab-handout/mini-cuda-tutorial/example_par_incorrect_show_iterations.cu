#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int *A, int *B, int *counter, int n) {
  int tid = threadIdx.x;
  
  if (tid < n) {
    for (int j=0; j<n; j++) {
      counter[tid*n+j]++;
      A[tid*n+j] = B[tid*n+j];
    }
  }
}

int main(int argc, char** argv)
{
  int i, j;

  if (argc < 2)
    exit (1);
  int n = atoi (argv[1]);

  int h_A[n][n];
  int h_B[n][n];
  int *d_A;
  int *d_B;

  int numBytes = n*n*sizeof(int);

  //allocate device memory
  cudaMalloc((void **)&d_A,numBytes);
  cudaMalloc((void **)&d_B,numBytes);

  //transfer data from host to device
  cudaMemcpy(d_A,h_A,numBytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,h_B,numBytes,cudaMemcpyHostToDevice);

  int h_counter[n][n];
  for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j)
      h_counter[i][j] = 0;

  int *d_counter;
  cudaMalloc((void **)&d_counter,numBytes);
  cudaMemcpy(d_counter,h_counter,numBytes,cudaMemcpyHostToDevice);

/*  #pragma omp parallel for
  for (i = 0; i < n; ++i)
    {
      for (j = 0; j < n; ++j)
	{
	  printf ("step: i=%d (%p) \t j=%d (%p)\n", i, &i, j, &j);
          #pragma omp atomic
	    counter[i][j]++;
	  A[i][j] = B[i][j];
	}
    }
*/

  kernel<<<n,n>>>(d_A,d_B,d_counter,n);

  //transfer data from device to host
  cudaMemcpy(h_A,d_A,numBytes,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B,d_B,numBytes,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_counter,d_counter,numBytes,cudaMemcpyDeviceToHost);


  printf ("=========================\n");
  int total_it = 0;
  printf ("i \\ j\t");
  for (j = 0; j < n; ++j)
    printf ("%d ", j);
  printf ("\n");
  for (i = 0; i < n; ++i)
    {
      printf ("%d\t", i);
      for (j = 0; j < n; ++j)
	{
	  printf ("%d ", h_counter[i][j]);
	  total_it += h_counter[i][j];
	}
      printf ("\n");
    }

  printf ("total iterations executed: %d (expected %d)\n", total_it, n * n);

  //free cuda memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_counter);

}
