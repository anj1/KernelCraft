#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Struct to hold all GPU resources (removed d_rho, d_ux, d_uy)
struct GPUResources {
    float *d_fluid;
    float *d_fluid_new;
    int *d_cxs;
    int *d_cys;
    float *d_weights;
    bool *d_obstacle;
    dim3 blockDim;
    dim3 gridDim;
};

// Transpose kernel for data layout transformation
__global__ void transposeKernel(float* dst, const float* src, int Nx, int Ny, int NL) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    for (int i = 0; i < NL; ++i) {
        dst[i * Nx * Ny + y * Nx + x] = src[y * Nx * NL + x * NL + i];
    }
}

// GPU kernel for calculating macroscopic variables
// Only used for testing, not part of the main simulation
__global__ void calculateMacroscopicKernel(float* fluid, float* rho, float* ux, float* uy,
                                         const int* cxs, const int* cys, int Nx, int Ny, int NL) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < Nx && y < Ny) {
        float local_rho = 0.0f;
        float local_ux = 0.0f;
        float local_uy = 0.0f;
        
        for (int i = 0; i < NL; i++) {
            float f = fluid[y * Nx * NL + x * NL + i];
            local_rho += f;
            local_ux += cxs[i] * f;
            local_uy += cys[i] * f;
        }
        
        rho[y * Nx + x] = local_rho;
        ux[y * Nx + x] = local_ux / local_rho;
        uy[y * Nx + x] = local_uy / local_rho;
    }
}

// Combined macroscopic and collision kernel
__global__ void macroAndCollisionKernel(
    float* fluid, float* fluid_new,
    const int* cxs, const int* cys, const float* weights,
    float tau, int Nx, int Ny, int NL) {

    __shared__ int s_cxs[9];
    __shared__ int s_cys[9];
    __shared__ float s_weights[9];

    int tix = threadIdx.x;
    if (tix < NL) {
        s_cxs[tix] = cxs[tix];
        s_cys[tix] = cys[tix];
        s_weights[tix] = weights[tix];
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx * NL + x * NL;

    // Compute macroscopic variables
    float rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for (int i = 0; i < NL; ++i) {
        float f = fluid[idx + i];
        rho += f;
        ux += s_cxs[i] * f;
        uy += s_cys[i] * f;
    }
    ux /= rho;
    uy /= rho;
    float usqr = ux * ux + uy * uy;

    // Collision step
    for (int i = 0; i < NL; ++i) {
        int idxf = idx + i;

        float cu = s_cxs[i] * ux + s_cys[i] * uy;
        float feq = rho * s_weights[i] * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usqr);
        fluid_new[idxf] = fluid[idxf] + (-1.0f/tau) * (fluid[idxf] - feq);
    }
}

// Optimized drift kernel with coalesced access
__global__ void driftKernel(
    float* dst, const float* src,
    const int* cxs, const int* cys,
    int Nx, int Ny, int NL) {

    __shared__ int s_cxs[9];
    __shared__ int s_cys[9];

    int tix = threadIdx.x;
    if (tix < NL) {
        s_cxs[tix] = cxs[tix];
        s_cys[tix] = cys[tix];
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    for (int i = 0; i < NL; ++i) {
        int xp = (x - s_cxs[i] + Nx) % Nx;
        int yp = (y - s_cys[i] + Ny) % Ny;
        dst[y * Nx * NL + x * NL + i] = src[yp * Nx * NL + xp * NL + i];
    }
}

// GPU kernel for bounce-back operation
__global__ void bounceBackKernel(float* fluid, float* fluid_orig, const bool* obstacle, int Nx, int Ny, int NL) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < Nx && y < Ny && obstacle[y * Nx + x]) {
        // Only process first half of directions (1 to (NL-1)/2)
        // Each thread handles the swap for one direction
        for (int i = 0; i < NL; i++) {
            int idx = y * Nx * NL + x * NL;

            //int opp = i + 4;  // Opposite direction index
            int opp = i==0 ? 0 : (((i - 1) + 4) % (NL - 1)) + 1;  // Opposite direction index

            fluid[idx + opp] = fluid_orig[idx + i];
        }
    }
}

// Initialization function with data layout transformation
extern "C" GPUResources* initialize_gpu(
    const float* h_fluid,
    const int* h_cxs,
    const int* h_cys,
    const float* h_weights,
    const bool* h_obstacle,
    int Nx,
    int Ny,
    int NL) {

    GPUResources* res = new GPUResources;
    size_t fluidSize = NL * Nx * Ny * sizeof(float);

    // Allocate device memory
    cudaMalloc(&res->d_fluid, fluidSize);
    cudaMalloc(&res->d_fluid_new, fluidSize);
    cudaMalloc(&res->d_cxs, NL * sizeof(int));
    cudaMalloc(&res->d_cys, NL * sizeof(int));
    cudaMalloc(&res->d_weights, NL * sizeof(float));
    cudaMalloc(&res->d_obstacle, Nx * Ny * sizeof(bool));

    if (res->d_fluid == NULL || res->d_fluid_new == NULL || res->d_cxs == NULL ||
        res->d_cys == NULL || res->d_weights == NULL || res->d_obstacle == NULL) {
        fprintf(stderr, "Error allocating device memory\n");
        return NULL;
    }

    // Copy and transpose fluid data to [i][y][x] layout
    float* temp_fluid;
    cudaMalloc(&temp_fluid, fluidSize);

    if (temp_fluid == NULL) {
        fprintf(stderr, "Error allocating device memory\n");
        return NULL;
    }

    cudaMemcpy(temp_fluid, h_fluid, fluidSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((Nx + 15)/16, (Ny + 15)/16);
    transposeKernel<<<grid, block>>>(res->d_fluid, temp_fluid, Nx, Ny, NL);
    cudaFree(temp_fluid);

    // Copy other data
    cudaMemcpy(res->d_cxs, h_cxs, NL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(res->d_cys, h_cys, NL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(res->d_weights, h_weights, NL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(res->d_obstacle, h_obstacle, Nx * Ny * sizeof(bool), cudaMemcpyHostToDevice);

    res->blockDim = dim3(16, 16);
    res->gridDim = dim3((Nx + 15)/16, (Ny + 15)/16);

    return res;
}

// Simulation step using optimized kernels
extern "C" void simulation_step(GPUResources* res, float tau, int Nx, int Ny, int NL) {
    size_t sharedMemSize = (2 * NL * sizeof(int)) + (NL * sizeof(float));

    // Drift step
    driftKernel<<<res->gridDim, res->blockDim, sharedMemSize>>>(
        res->d_fluid_new, res->d_fluid, res->d_cxs, res->d_cys, Nx, Ny, NL);

    // Swap pointers
    std::swap(res->d_fluid, res->d_fluid_new);

    // Combined macro and collision
    macroAndCollisionKernel<<<res->gridDim, res->blockDim, sharedMemSize>>>(
        res->d_fluid, res->d_fluid_new, res->d_cxs, res->d_cys, res->d_weights,
        tau, Nx, Ny, NL);

    // Swap pointers again
    std::swap(res->d_fluid, res->d_fluid_new);

    // Bounce-back
    bounceBackKernel<<<res->gridDim, res->blockDim>>>(
         res->d_fluid, res->d_fluid_new, res->d_obstacle, Nx, Ny, NL);
}

// Function to clean up GPU resources
extern "C" void cleanup_gpu(GPUResources* res) {
    if (res) {
        cudaFree(res->d_fluid);
        cudaFree(res->d_fluid_new);
        cudaFree(res->d_cxs);
        cudaFree(res->d_cys);
        cudaFree(res->d_weights);
        cudaFree(res->d_obstacle);
        delete res;
    }
}

extern "C" void set_fluid(
    GPUResources* res,
    float* h_fluid,
    int Nx,
    int Ny,
    int NL) {
    // Update fluid state on device
    cudaMemcpy(res->d_fluid, h_fluid, Nx * Ny * NL * sizeof(float), cudaMemcpyHostToDevice);
}

extern "C" void get_macroscopic(
    GPUResources* res,
    float* h_rho,
    float* h_ux,
    float* h_uy,
    int Nx,
    int Ny,
    int NL) {

    float *d_rho, *d_ux, *d_uy;

    cudaMalloc(&d_rho, Nx * Ny * sizeof(float));
    cudaMalloc(&d_ux,  Nx * Ny * sizeof(float));
    cudaMalloc(&d_uy,  Nx * Ny * sizeof(float));

    if (d_rho == NULL || d_ux == NULL || d_uy == NULL) {
        fprintf(stderr, "Error allocating device memory\n");
        return;
    }

    calculateMacroscopicKernel<<<res->gridDim, res->blockDim>>>(
        res->d_fluid, d_rho, d_ux, d_uy,
        res->d_cxs, res->d_cys, Nx, Ny, NL);

    cudaDeviceSynchronize();

    cudaMemcpy(h_rho, d_rho, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux,  d_ux, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy,  d_uy, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
}
