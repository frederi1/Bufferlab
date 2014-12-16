#include <stdio.h>

__device__ int blockSum(int *b, int size) {
  int sum=0, i;
  for (i=0; i<size;++i) {
    sum += b[i];
  }
  return sum;
}

// Compute the sum of each subblock and write the result to the first
// index in "a" where the subblock starts. For this code to
// work, Number of blocks * Number of threads = N. No more no less
__global__ void computeSumPerBlock(int *a, int N) {
  //each block has its own sdata_a shared memory area
  extern __shared__ int sdata_a[];
  int tmp;

  //each thread loads 1 element from global to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i<N) {
    sdata_a[tid] = a[i];
    // Make sure we load all values of a to shared memory before
    //compute the sum of each subblock.
    __syncthreads();

    // All blocks execute this in parallel. Note each block has its own
    //shared memory sdata_a.
    if (tid == 0) {
      tmp = blockSum(sdata_a,blockDim.x);
      a[i] = tmp;
    }
  }
}

void printResultArray(int *a, int N) {
  int i;
  for (i=0;i<N;++i) {
    printf("%d  ",a[i]);
  }
  printf("\n");
}

int main() {
  //allocate host memory
  int N = 10;
  int h_a[] = {1,2,3,4,5,6,7,8,9,10};

  //allocate device memory
  int *d_a;
  cudaMalloc((void **)&d_a, N*sizeof(int));

  //copy data from host to device
  cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

  // 	Case 1: We use 2 blocks, 5 threads per block. Thus, given array 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
  // 	  We compute sum of first block which is 1+2+3+4+5=15 and write it to h_a[0]
  // 	  We compute sum of second block which is 6+7+8+9+10=40 and write it to h_a[5]
  // 	  Note we do these sums in parallel
  //           The expected output is 15, 2, 3, 4, 5, 40, 7, 8, 9, 10

  //execute the kernel
  computeSumPerBlock<<<2,5>>>(d_a, N);

  //copy data from device back to host
  cudaMemcpy(h_a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);

  //print the result
  printf("Case 1: 2 blocks, 5 threads per block\n");
  printResultArray(h_a,N);

  // 	Case 2: We use 5 blocks, 2 threads per block. Thus, given array 1, 2, 3, 4, 5, 6, 7, 8, 9 ,10
  // 	  We compute sum of first block which is 1+2=3 and write it to h_b[0]
  // 	  We compute sum of second block which is 3+4=7 and write it to h_b[2]
  // 	  We compute sum of third block which is 5+6=11 and write it to h_b[4]
  // 	  We compute sum of fourth block which is 7+8=15 and write it to h_b[6]
  // 	  We compute sum of fifth block which is 9+10=19 and write it to h_b[8]
  // 	  Note we do these sums in parallel
  //           The expected output is 3, 2, 7, 4, 11, 6, 15, 8, 19, 10
  int h_b[] = {1,2,3,4,5,6,7,8,9,10};

  //copy data from host to device
  cudaMemcpy(d_a, h_b, N*sizeof(int), cudaMemcpyHostToDevice);

  //execute the kernel
  computeSumPerBlock<<<5,2>>>(d_a, N);

  //copy data from device back to host
  cudaMemcpy(h_b, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);

  //print the result
  printf("Case 2: 5 blocks, 2 threads per block\n");
  printResultArray(h_b,N);

  //free device memory
  cudaFree(d_a);
}
