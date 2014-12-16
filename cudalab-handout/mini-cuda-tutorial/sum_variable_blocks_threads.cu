#include <stdio.h>

__global__ void sum(int *a, int *b, int *c, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid<N) {
    c[tid] = a[tid] + b[tid];
  }
}

//host function
void initialize(int *a, int c, int N) {
  int i;
  for (i=0;i<N;++i) {
    a[i] = c;
  }
}

void printResultArray(int *c, int N) {
  int i;
  for (i=0;i<N;++i) {
    printf("%d  ",c[i]);
  }
  printf("\n");
}

int main() {
  //allocate host memory
  int N = 10;
  int *h_a = (int *)malloc(N*sizeof(int));
  int *h_b = (int *)malloc(N*sizeof(int));
  int *h_c = (int *)malloc(N*sizeof(int));

  //intialize h_a to all 1s
  initialize(h_a,1,N);
  initialize(h_b,5,N);

  //allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, N*sizeof(int));
  cudaMalloc((void **)&d_b, N*sizeof(int));
  cudaMalloc((void **)&d_c, N*sizeof(int));

  //copy data from host to device
  cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);

  // Case 1: Execute the kernel with 1 block and 1 thread. With this
  // case, there is only 1 thread with global id = 0 and thus we only do
  // addition on the first element of the array.
  // The expected output is therefore: 6 0 0 0 0 0 0 0 0 0

  //execute the kernel
  sum<<<1,1>>>(d_a, d_b, d_c, N);

  //copy data from device back to host
  cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  //print the result
  printf("Case 1: 1 block, 1 thread per block\n");
  printResultArray(h_c,N);

  // Case 2: Execute the kernel with 1 block and 4 threads. With this
  // case, there are 4 threads with global id = 0, 1, 2, 3 therefore,
  // the device only perform additions on the first 4 elements of the
  // array.
  // The expected output is therefore: 6 6 6 6 0 0 0 0 0 0

  // reset the d_c array
  cudaMemset(d_c, 0, N*sizeof(int));
  sum<<<1,4>>>(d_a, d_b, d_c, N);

  //copy data from device back to host
  cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  //print the result
  printf("Case 2: 1 block, 4 threads per block\n");
  printResultArray(h_c,N);

   // Case 3: Execute the kernel with 4 blocks and 2 thread per
   // block. With this case, we have total 4 threads with global id =
   // 0, 1, 2, 3, 4, 5, 6, 7
  // Therefore, the device only performs additions on the first 8
  // elements of the array.
  // The expected output is therefore: 6 6 6 6 6 6 6 6 0 0

  // reset the d_c array
  cudaMemset(d_c, 0, N*sizeof(int));
  sum<<<4,2>>>(d_a, d_b, d_c, N);

  //copy data from device back to host
  cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  //print the result
  printf("Case 3: 4 blocks, 2 threads per block\n");
  printResultArray(h_c,N);

  // Case 4: To fully do additions on every elements of the array, we
  // must have >= N threads. For example, with 2 blocks and 10 threads
  // per block, we will have total 20 threads which are more than
  // enough to operate this kernel. Since we have the check tid<N in
  // the kernel, we infact only use N threads to compute the array sum
  // despite the fact that we have more than number of threads needed.
  // The expected output is therefore: 6 6 6 6 6 6 6 6 6 6

  //reset the d_c array
  cudaMemset(d_c, 0, N*sizeof(int));
  sum<<<2,10>>>(d_a, d_b, d_c, N);

  //copy data from device back to host
  cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  //print the result
  printf("Case 4: 2 blocks, 10 threads per block\n");
  printResultArray(h_c,N);

  //free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  //free host memory
  free(h_a);
  free(h_b);
  free(h_c);
}
