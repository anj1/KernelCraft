#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

template <int N>
__device__ __forceinline__ int swap_idx(int i, int j) {
    return ((N-1)*i + (N-1)*j + 1) % N;
}

template <int N, int N_GRID>
__device__ void warpShuffleRow2Col(float *row, int wid, int bid)
{
    const int W = N*N;
    const int N_BITS = W*N_GRID;
    const uint32_t MASK = (uint32_t)((((uint64_t)1) << N_BITS) - 1);

    int wm3 = wid % N;
    int w3 = wid / N;

    #pragma unroll
    for (int j=0; j<N; j++) {
        #pragma unroll
        for (int i=0; i<N; i++) {
            int idx = swap_idx<N>(i, wm3) + N*swap_idx<N>(j, w3);
            row[idx] = __shfl_sync(MASK, row[idx], idx + W*bid);
        }
    }

    __syncwarp();
}

template <int N, int N_GRID>
__device__ void warpShuffleRow2Box(float *row, int wid, int bid)
{
    const int W = N*N;
    const int N_BITS = W*N_GRID;
    const uint32_t MASK = (uint32_t)((((uint64_t)1) << N_BITS) - 1);

    int wm3 = wid % N;
    int w3 = wid / N;

    int nj = N == 2 ? 1 : N;
    #pragma unroll
    for(int j = 0; j < nj; j++){ 
        int src = N*w3 + swap_idx<N>(N-j,wm3) + W*bid;
        #pragma unroll
        for(int i=0; i < N; i++) {
            int idx = N*swap_idx<N>(N-j, wm3) + i;
            row[idx] = __shfl_sync(MASK, row[idx], src);
        }
    }

    __syncwarp();
}

// Kernel for NxN matrix transpose using warp shuffling
template <int N, int N_GRID>
__global__ void warpShuffleNxNTranspose(float* matrix, int transposeOp) {
    const int W = N*N;

    // Each warp handles a NxN tile
    // Thread indices within the warp
    int wid = threadIdx.x % W;
    int bid = threadIdx.x / W;
    
    // Load elements per thread
    float row[W];

    // Copy from global memory to local memory
    #pragma unroll
    for (int i = 0; i < W; i++) {
        row[i] = matrix[W*W*bid + W*wid + i];
    }

    if (transposeOp == 1) {
        warpShuffleRow2Col<N, N_GRID>(row, wid, bid);
    } else if (transposeOp == 2) {
        warpShuffleRow2Box<N, N_GRID>(row, wid, bid);
    }

    // Copy from local memory to global memory
    #pragma unroll
    for (int i = 0; i < W; i++) {
        matrix[W*W*bid + W*wid + i] = row[i];
    }
}

// Host function to launch the kernel
template <int N, int N_GRID>
void transposeNxNMatrix(float* d_matrix, int transposeOp) {
    dim3 block(32, 1);
    dim3 grid(1, 1);
    
    warpShuffleNxNTranspose<N, N_GRID><<<grid, block>>>(d_matrix, transposeOp);
}

float test_data(int size, int k, int i, int j)
{
    int im = i / size;
    int jm = j / size;

    if (im == jm) {
        //int idx = 32 * i + j;
        //return (float)(idx % 7);
        return size*j + i + ((float)k)/100;
    } else {
        return 0.0;
    }
}

constexpr int N_GRIDS[6] = {0, 0, 4, 3, 2, 1};

// transposeOp:
// 0: none
// 1: transpose
// 2: box
bool test(int N, int transposeOp){
    int W = N*N;
    const int N_GRID = N_GRIDS[N];

    const int SIZE = W * W * N_GRID;
    float* h_matrix;
    float* d_matrix;
    
    // Allocate host memory
    h_matrix = (float*)malloc(W*W*N_GRID*sizeof(float));
    
    // Initialize matrix with test data
    for(int k=0; k<N_GRID; k++) {
        for (int j = 0; j < W; j++) {
            for (int i = 0; i < W; i++) {
                h_matrix[W*W*k + W*j + i] = test_data(W, k, i, j);
            }
        }
    }

    // print the matrix
    for(int k=0; k<N_GRID; k++) {
        for (int i = 0; i < W*W; i++) {
            printf("% 2.3f ", h_matrix[W*W*k + i]);
            if ((i + 1) % W == 0) {
                printf("\n");
            }
        }
        printf("\n");
    }
    
    // Allocate device memory
    cudaMalloc(&d_matrix, SIZE * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_matrix, h_matrix, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform transpose
    switch (N) {
        case 2:
            transposeNxNMatrix<2, N_GRIDS[2]>(d_matrix, transposeOp);
            break;
        case 3:
            transposeNxNMatrix<3, N_GRIDS[3]>(d_matrix, transposeOp);
            break;
        case 4:
            transposeNxNMatrix<4, N_GRIDS[4]>(d_matrix, transposeOp);
            break;
        case 5:
            transposeNxNMatrix<5, N_GRIDS[5]>(d_matrix, transposeOp);
            break;
        default:
            printf("Unsupported matrix size\n");
            return 1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_matrix, d_matrix, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // ANSI color codes
    const char* RESET = "\033[0m";
    const char* RED = "\033[31m";
    const char* GREEN = "\033[32m";

    // print the matrix, showing the incorrect elements in red
    int diff_count = 0;
    for(int k=0; k<N_GRID; k++) {
        for (int j = 0; j < W; j++) {
            for (int i = 0; i < W; i++) {
                float v = h_matrix[W*W*k + W*j + i];
                float ref;
                switch(transposeOp) {
                    case 0:
                        ref = test_data(W, k, i, j);
                        break;
                    case 1:
                        ref = test_data(W, k, j, i);
                        break;
                    case 2:
                        int i3 = i / N;
                        int im = i % N;
                        int j3 = j / N;
                        int jm = j % N;
                        ref = test_data(W, k, im + N*jm, i3 + N*j3);
                        break;
                };

                if (v != ref) {
                    printf("%s% 2.3f%s ", RED, v, RESET);
                    diff_count++;
                } else {
                    printf("%s% 2.3f%s ", GREEN, v, RESET);
                }
                if ((i + 1) % W == 0) {
                    printf("\n");
                }
            }
        }
        printf("\n");
    }

    // Clean up
    free(h_matrix);
    cudaFree(d_matrix);

    if (diff_count == 0) {
        printf("Matrix transposed successfully\n");
        return true;
    } else {
        printf("Matrix transpose failed; %d elements differ\n", diff_count);
        return false;
    }

    return 0;
}

// Example usage
int main() {
    bool success = true;

    success &= test(2, 1);
    success &= test(2, 2);

    success &= test(3, 1);
    success &= test(3, 2);

    success &= test(4, 1);
    success &= test(4, 2);

    success &= test(5, 1);
    success &= test(5, 2);

    if (success) {
        printf("All tests passed\n");
    } else {
        printf("Some tests failed\n");
    }

    return 0;
}