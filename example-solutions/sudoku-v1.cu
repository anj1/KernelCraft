#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>


// Constants for GPU configuration
const int BLOCK_SIZE = 256;
const int MAX_PROPOSALS_PER_BLOCK = 128;  // Can be tuned based on shared memory size
const int GRID_SIZE = 9;
const int GRID_N = GRID_SIZE * GRID_SIZE;
const int SUBGRID_SIZE = 3;

using cell_t = char;

// Structure to hold a Sudoku grid
struct SudokuGrid {
    cell_t cells[GRID_N];  // TODO: char?
    int numFilled;  // Track number of filled cells
};


// Device function to check if a row is valid
__device__ bool isRowValid(const cell_t grid[GRID_N], int row, bool used[]) {
    // Reset used array
    for (int i = 0; i < GRID_SIZE; i++) {
        used[i] = false;
    }
    
    // Check row
    for (int col = 0; col < GRID_SIZE; col++) {
        cell_t val = grid[row*GRID_SIZE + col];
        if (val != 0) {
            if (used[val-1]) return false;
            used[val-1] = true;
        }
    }
    return true;
}

// Device function to check if a column is valid
__device__ bool isColValid(const cell_t grid[GRID_N], int col, bool used[]) {
    // Reset used array
    for (int i = 0; i < GRID_SIZE; i++) {
        used[i] = false;
    }
    
    // Check column
    for (int row = 0; row < GRID_SIZE; row++) {
        cell_t val = grid[row*GRID_SIZE+col];
        if (val != 0) {
            if (used[val-1]) return false;
            used[val-1] = true;
        }
    }
    return true;
}

// Device function to check if a 3x3 subgrid is valid
__device__ bool isSubgridValid(const cell_t grid[GRID_N], int startRow, int startCol, bool used[]) {
    // Reset used array
    for (int i = 0; i < GRID_SIZE; i++) {
        used[i] = false;
    }
    
    // Check 3x3 subgrid
    for (int i = 0; i < SUBGRID_SIZE; i++) {
        for (int j = 0; j < SUBGRID_SIZE; j++) {
            cell_t val = grid[(startRow + i)*GRID_SIZE + startCol + j];
            if (val != 0) {
                if (used[val-1]) return false;
                used[val-1] = true;
            }
        }
    }
    return true;
}

// Device function to find next empty cell
__device__ bool findEmptyCell(const cell_t grid[GRID_N], int& row, int& col) {
    for (row = 0; row < GRID_SIZE; row++) {
        for (col = 0; col < GRID_SIZE; col++) {
            if (grid[row*GRID_SIZE+col] == 0) return true;
        }
    }
    return false;
}

// Kernel 1: Check validity of proposals
__global__ void checkValidityKernel(SudokuGrid* proposals, bool* validFlags, int numProposals) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalId = bid * blockDim.x + tid;
    
    if (globalId >= numProposals) return;
    
    bool used[GRID_SIZE];
    bool isValid = true;
    SudokuGrid currentGrid = proposals[globalId];
    
    // Check all rows
    for (int row = 0; row < GRID_SIZE && isValid; row++) {
        isValid = isRowValid(currentGrid.cells, row, used);
    }
    
    // Check all columns
    for (int col = 0; col < GRID_SIZE && isValid; col++) {
        isValid = isColValid(currentGrid.cells, col, used);
    }
    
    // Check all 3x3 subgrids
    for (int i = 0; i < GRID_SIZE && isValid; i += SUBGRID_SIZE) {
        for (int j = 0; j < GRID_SIZE && isValid; j += SUBGRID_SIZE) {
            isValid = isSubgridValid(currentGrid.cells, i, j, used);
        }
    }
    
    validFlags[globalId] = isValid;
}

// Kernel 2: Generate new proposals
__global__ void generateProposalsKernel(SudokuGrid* validProposals, int numValidProposals,
                                       SudokuGrid* newProposals, int* newProposalCount,
                                       bool* foundSolution, SudokuGrid* solution) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalId = bid * blockDim.x + tid;
    
    if (globalId >= numValidProposals) return;
    
    SudokuGrid currentGrid = validProposals[globalId];
    
    // Check if we found a solution
    if (currentGrid.numFilled == GRID_N) {
        *foundSolution = true;
        *solution = currentGrid;
        return;
    }
    
    // Find next empty cell
    int row, col;
    if (findEmptyCell(currentGrid.cells, row, col)) {
        // Try each possible number
        for (cell_t num = 1; num <= GRID_SIZE; num++) {
            SudokuGrid newProposal = currentGrid;
            newProposal.cells[row*GRID_SIZE + col] = num;
            newProposal.numFilled++;
            
            // Add new proposal to global list
            int newIdx = atomicAdd(newProposalCount, 1);
            newProposals[newIdx] = newProposal;
        }
    }
}

// Modified host function
extern "C" bool solveSudokuGPU(SudokuGrid& initial, SudokuGrid& solution) {
    // Allocate device memory
    size_t maxProposals = 1<<22;
    
    thrust::device_vector<SudokuGrid> d_proposals(maxProposals);
    thrust::device_vector<SudokuGrid> d_newProposals(maxProposals);
    thrust::device_vector<bool> d_validFlags(maxProposals);
    
    SudokuGrid *d_solution;
    bool *d_foundSolution;
    int *d_newProposalCount;
    
    cudaMalloc(&d_solution, sizeof(SudokuGrid));
    cudaMalloc(&d_foundSolution, sizeof(bool));
    cudaMalloc(&d_newProposalCount, sizeof(int));
    
    // Copy initial grid to device
    d_proposals[0] = initial;
    
    bool foundSolution = false;
    int numProposals = 1;
    
    while (!foundSolution && numProposals > 0) {
        //std::cout << "Number of proposals: " << numProposals << std::endl;
        
        // Step 1: Check validity
        int numBlocks = (numProposals + BLOCK_SIZE - 1) / BLOCK_SIZE;
        checkValidityKernel<<<numBlocks, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(d_proposals.data()),
            thrust::raw_pointer_cast(d_validFlags.data()),
            numProposals
        );
        
        // Step 2: Compact valid proposals using Thrust
        thrust::device_vector<SudokuGrid> d_validProposals(numProposals);
        int numValidProposals = thrust::copy_if(
            d_proposals.begin(),
            d_proposals.begin() + numProposals,
            d_validFlags.begin(),
            d_validProposals.begin(),
            thrust::identity<bool>()
        ) - d_validProposals.begin();
        
        // Step 3: Generate new proposals
        int newProposalCount = 0;
        cudaMemcpy(d_newProposalCount, &newProposalCount, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_foundSolution, &foundSolution, sizeof(bool), cudaMemcpyHostToDevice);
        
        numBlocks = (numValidProposals + BLOCK_SIZE - 1) / BLOCK_SIZE;
        generateProposalsKernel<<<numBlocks, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(d_validProposals.data()),
            numValidProposals,
            thrust::raw_pointer_cast(d_newProposals.data()),
            d_newProposalCount,
            d_foundSolution,
            d_solution
        );
        
        // Check if solution was found
        cudaMemcpy(&foundSolution, d_foundSolution, sizeof(bool), cudaMemcpyDeviceToHost);
        if (foundSolution) {
            cudaMemcpy(&solution, d_solution, sizeof(SudokuGrid), cudaMemcpyDeviceToHost);
            break;
        }
        
        // Update proposals for next iteration
        cudaMemcpy(&numProposals, d_newProposalCount, sizeof(int), cudaMemcpyDeviceToHost);
        thrust::copy(d_newProposals.begin(), d_newProposals.begin() + numProposals, d_proposals.begin());
    }
    
    // Clean up
    cudaFree(d_solution);
    cudaFree(d_foundSolution);
    cudaFree(d_newProposalCount);
    
    return foundSolution;
}

// Your existing main function and getNumFilled function remain the same...

int getNumFilled(const cell_t grid[GRID_N]) {
    int count = 0;
    for (int i = 0; i < GRID_N; i++) {
        if (grid[i] != 0) count++;
    }
    return count;
}

// Example usage
int main() {
    SudokuGrid initial = {
        // Daily Telegraph January 19th "Diabolical"
        // {
        //     0,2,0, 6,0,8, 0,0,0,
        //     5,8,0, 0,0,9, 7,0,0,
        //     0,0,0, 0,4,0, 0,0,0,

        //     3,7,0, 0,0,0, 5,0,0,
        //     6,0,0, 0,0,0, 0,0,4,
        //     0,0,8, 0,0,0, 0,1,3,

        //     0,0,0, 0,2,0, 0,0,0,
        //     0,0,9, 8,0,0, 0,3,6,
        //     0,0,0, 3,0,6, 0,9,0
        // },

        // Challenge 2 from Sudoku Solver by Logic
        // {
        //     2,0,0, 3,0,0, 0,0,0,
        //     8,0,4, 0,6,2, 0,0,3,
        //     0,1,3, 8,0,0, 2,0,0,

        //     0,0,0, 0,2,0, 3,9,0,
        //     5,0,7, 0,0,0, 6,2,1,
        //     0,3,2, 0,0,6, 0,0,0,

        //     0,2,0, 0,0,9, 1,4,0,
        //     6,0,1, 2,5,0, 8,0,9,
        //     0,0,0, 0,0,1, 0,0,2
        // },

        // Not fun 
        // {
        //     0,2,0, 0,0,0, 0,0,0,
        //     0,0,0, 6,0,0, 0,0,3,
        //     0,7,4, 0,8,0, 0,0,0,

        //     0,0,0, 0,0,3, 0,0,2,
        //     0,8,0, 0,4,0, 0,1,0,
        //     6,0,0, 5,0,0, 0,0,0,

        //     0,0,0, 0,1,0, 7,8,0,
        //     5,0,0, 0,0,9, 0,0,0,
        //     0,0,0, 0,0,0, 0,4,0
        // },

        // Arto Inkala's "Hardest-Ever Sudoku"
        {
            8,0,0, 0,0,0, 0,0,0,
            0,0,3, 6,0,0, 0,0,0,
            0,7,0, 0,9,0, 2,0,0,

            0,5,0, 0,0,7, 0,0,0,
            0,0,0, 0,4,5, 7,0,0,
            0,0,0, 1,0,0, 0,3,0,

            0,0,1, 0,0,0, 0,6,8,
            0,0,8, 5,0,0, 0,1,0,
            0,9,0, 0,0,0, 4,0,0
        },

        0  // Initial number of filled cells
    };
    initial.numFilled = getNumFilled(initial.cells);
    
    SudokuGrid solution;
    // time the solution
    auto start = std::chrono::high_resolution_clock::now();
    bool solved = solveSudokuGPU(initial, solution);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    if (solved) {
        printf("Solution found in %.3f seconds:\n", elapsed.count());
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                printf("%d ", solution.cells[i*GRID_SIZE + j]);
            }
            printf("\n");
        }
    } else {
        printf("No solution found.\n");
    }
    
    return 0;
}