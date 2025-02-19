#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <chrono>

#include <iostream>
#include <fstream>
#include <string>
#include <cctype>

// Constants for different Sudoku sizes
const int MAX_POSSIBILITIES = 32; // Using 32-bit masks
const int MAX_THREADS = 5 * 1024;
const int MAX_PROPOSALS = 512 * 1024; // Maximum number of proposals in global memory
const int BLOCK_SIZE = 256;

#define FIXED_BIT (1u << 31)

void printSudoku(char grid[], int boxSize);

template <int SIZE>
struct SudokuCandidates {
    uint32_t cells[SIZE * SIZE];  // Possibility masks for each cell
};
template <int SIZE>
struct SudokuProposal {
    char cells[SIZE * SIZE];  // Fixed values 
};

__device__ int popcount(uint32_t x) {
    return __popc(x);
}

struct CellVote {
    int cell_idx;
    int possibility_count;
    
    __device__ static unsigned long long pack(int count, int idx) {
        return (static_cast<unsigned long long>(count) << 32) | static_cast<unsigned int>(idx);
    }
    
    __device__ static void unpack(unsigned long long packed, int& count, int& idx) {
        count = static_cast<int>(packed >> 32);
        idx = static_cast<int>(packed & 0xFFFFFFFF);
    }
};

template <int SIZE>
__device__ CellVote findLeastCandidates(const SudokuCandidates<SIZE>& prop) {
    CellVote vote = {-1, MAX_POSSIBILITIES};
    
    for (int i = 0; i < SIZE * SIZE; i++) {
        if (prop.cells[i] == 0){ // Invalid proposal
            vote.cell_idx = -1;
            return vote;
        }
        int count = popcount(prop.cells[i] & ~FIXED_BIT);
        if ((count > 1) && (count < vote.possibility_count)) {
            vote.possibility_count = count;
            vote.cell_idx = i;
        }
    }

    return vote;
}

template <int SIZE>
__device__ CellVote findLeastCandidatesLocal(const uint32_t* shared_cells, int tid, int *shared_count, int *shared_idx) {
    
    // Each thread checks its row
    int best_count = MAX_POSSIBILITIES;
    int best_idx = -1;
    
    for (int i = 0; i < SIZE; i++) {
        int idx = tid * SIZE + i;
        uint32_t cell = shared_cells[idx];
        
        if (cell == 0) { // Invalid cell
            best_idx = -2;
            best_count = 0; // Invalid proposal
            break;
        }
        
        if (!(cell & FIXED_BIT)) {
            int count = popcount(cell);
            if (count > 1 && count < best_count) {
                best_count = count;
                best_idx = idx;
            }
        }
    }
    
    if (tid == 0) {
        *shared_count = MAX_POSSIBILITIES;
        *shared_idx = -1;
    }
    __syncthreads();
    
    // If this thread found a valid candidate, participate in the atomic minimum
    if (best_idx != -1) {
        atomicMin(shared_count, best_count);

        if (*shared_count == best_count) {
            // If we won the race, store our index
            *shared_idx = best_idx;
        }
    }
    __syncthreads();
}

template <int SIZE>
__device__ bool isSolution(const  SudokuProposal<SIZE>& prop) {
    for (int i = 0; i < SIZE * SIZE; i++) {
        if (prop.cells[i] == 0) {
            return false;
        }
    }
    return true;
}

template<int N> // N=3 for 9x9, N=4 for 16x16
__device__ int updateGrpCandidt(
    uint32_t* cells,
    int offs,       // offset into cells array
    int pi,         // pitch for i direction
    int pj          // pitch for j direction
) {
    int status = 0;

    // Get all values that appear as singles in any cell
    uint32_t singles = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            auto idx = offs + pi*i + pj*j;
            if (cells[idx] & FIXED_BIT) {
                auto mask = cells[idx] & ~FIXED_BIT;
                // Check if this value is already present in singles
                singles |= (singles & mask) ? FIXED_BIT : mask;
            }
        }
    }

    if(singles & FIXED_BIT) {
        // Invalid puzzle state. Mark cell as invalid and clear fixed bit
        cells[0] = 0;
        status = 3;
    }

    // Remove all singles from non-fixed cells in one operation
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            auto idx = offs + pi*i + pj*j;
            auto cell = cells[idx];

            if (!(cell & FIXED_BIT)) {
                uint32_t old_cell = cell;
                cell &= ~singles;
                if (popcount(cell) == 1) {
                    cell |= FIXED_BIT;
                }
                cells[idx] = cell;
                status |= (old_cell != cell);
            }
        }
    }
    return status;
}

template<int N>  // N=3 for 9x9, N=4 for 16x16
__device__ int removeNakedPairs(
    uint32_t* cells,
    int offs,       // offset into cells array
    int pi,         // pitch for i direction
    int pj          // pitch for j direction
) {
    uint32_t pairs[32] = {0};  // Track potential pairs
    uint32_t pair_mask = 0;    // Accumulated mask of values to remove
    int status = 0;

    char popcounts[N*N];

    // First pass: Find pairs
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            auto idx = offs + pi*i + pj*j;
            uint32_t cell = cells[idx];
            
            popcounts[N*i + j] = popcount(cell);
            if (!(cell & FIXED_BIT) && popcounts[N*i + j] == 2) {
                // Find the two set bits using ffs
                //uint32_t b0 = __ffs(cell) - 1;
                uint32_t b0 = 31 - __clz(cell);
                uint32_t bm = 1u << b0;
                uint32_t cellx = cell ^ bm;  // Clear first bit
                //uint32_t b1 = __ffs(cellx) - 1;
                uint32_t b1 = 31 - __clz(cellx);
                
                // Check if we've seen this pair before
                if (pairs[b1] & bm) {
                    // Found a naked pair!
                    pair_mask |= cell;
                } else {
                    // Record this potential pair
                    pairs[b1] |= bm;
                }
            }
        }
    }
    
    // Second pass: Remove pair values from other cells
    if (pair_mask) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                auto idx = offs + pi*i + pj*j;
                uint32_t cell = cells[idx];
                
                // check to make sure we're not removing the pair itself
                bool isCellPair = (!(cell & FIXED_BIT)) && (popcounts[N*i + j] == 2);
                bool isNakdPair = (cell & pair_mask) == cell;

                if (!(isCellPair && isNakdPair)) {
                    uint32_t old_cell = cell;
                    cell &= ~pair_mask;
                    if (popcount(cell) == 1) {
                        cell |= FIXED_BIT;
                    }
                    cells[idx] = cell;
                    status |= (old_cell != cell);
                }
            }
        }
    }

    return status;
}

template<int N>  // N=3 for 9x9, N=4 for 16x16
__device__ void updateCandidates(uint32_t* cells) {
    const int SIZE = N * N;
    int status;
    
    //do {
    for(int kk = 0; kk < N; kk++) {
        status = 0;
        
        // Update rows
        for (int row = 0; row < SIZE; row++) {
            status |= updateGrpCandidt<N>(cells, row*SIZE, 1, N);
            status |= removeNakedPairs<N>(cells, row*SIZE, 1, N);
        }
        
        // Check if state is invalid
        if (status==3) {
            return;
        }

        // Update columns
        for (int col = 0; col < SIZE; col++) {
            status |= updateGrpCandidt<N>(cells, col, SIZE, SIZE*N);
            status |= removeNakedPairs<N>(cells, col, SIZE, SIZE*N);
        }

        if (status==3) {
            return;
        }
        
        // Update boxes
        for (int box_row = 0; box_row < SIZE; box_row += N) {
            for (int box_col = 0; box_col < SIZE; box_col += N) {
                status |= updateGrpCandidt<N>(cells, box_row * SIZE + box_col, 1, SIZE);
                status |= removeNakedPairs<N>(cells, box_row * SIZE + box_col, 1, SIZE);
            }
        }
        
        // This doesn't actually speed things up much
        if (status==3) {
           return;
        }

    }
    //} while (status);  // Repeat until no more changes
}

template<int N>  // N=3 for 9x9, N=4 for 16x16
__device__ int updateGrpCandidtLocal(
    uint32_t* local_cells  // Local array for thread
) {
    int status = 0;

    // Get all values that appear as singles in any cell
    uint32_t singles = 0;
    for (int i = 0; i < N*N; i++) {
        if (local_cells[i] & FIXED_BIT) {
            auto mask = local_cells[i] & ~FIXED_BIT;
            singles |= (singles & mask) ? FIXED_BIT : mask;
        }
    }

    if (singles & FIXED_BIT) {
        local_cells[0] = 0;
        return 3;
    }

    // Remove all singles from non-fixed cells
    for (int i = 0; i < N*N; i++) {
        if (!(local_cells[i] & FIXED_BIT)) {
            uint32_t old_cell = local_cells[i];
            local_cells[i] &= ~singles;
            if (popcount(local_cells[i]) == 1) {
                local_cells[i] |= FIXED_BIT;
            }
            status |= (old_cell != local_cells[i]);
        }
    }

    return status;
}

template<int N>
__device__ int removeNakedPairsLocal(
    uint32_t* local_cells  // Local array for thread
) {
    uint32_t pairs[32] = {0};
    uint32_t pair_mask = 0;
    char popcounts[N*N];
    int status = 0;

    // Find pairs
    for (int i = 0; i < N*N; i++) {
        popcounts[i] = popcount(local_cells[i]);
        if (!(local_cells[i] & FIXED_BIT) && popcounts[i] == 2) {
            uint32_t b0 = 31 - __clz(local_cells[i]);
            uint32_t bm = 1u << b0;
            uint32_t cellx = local_cells[i] ^ bm;
            uint32_t b1 = 31 - __clz(cellx);
            
            if (pairs[b1] & bm) {
                pair_mask |= local_cells[i];
            } else {
                pairs[b1] |= bm;
            }
        }
    }
    
    // Remove pair values
    if (pair_mask) {
        for (int i = 0; i < N*N; i++) {
            bool isCellPair = (!(local_cells[i] & FIXED_BIT)) && (popcounts[i] == 2);
            bool isNakdPair = (local_cells[i] & pair_mask) == local_cells[i];

            if (!(isCellPair && isNakdPair)) {
                uint32_t old_cell = local_cells[i];
                local_cells[i] &= ~pair_mask;
                if (popcount(local_cells[i]) == 1) {
                    local_cells[i] |= FIXED_BIT;
                }
                status |= (old_cell != local_cells[i]);
            }
        }
    }

    return status;
}

template<int N>
__device__ void loadGroup(
    uint32_t* shared_cells,
    uint32_t* local_cells,
    int offs,      // offset into cells array
    int pi,        // pitch for i direction
    int pj         // pitch for j direction
) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            local_cells[i*N + j] = shared_cells[offs + pi*i + pj*j];
        }
    }
}

template<int N>
__device__ void storeGroup(
    uint32_t* shared_cells,
    uint32_t* local_cells,
    int offs,      // offset into cells array
    int pi,        // pitch for i direction
    int pj         // pitch for j direction
) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            shared_cells[offs + pi*i + pj*j] = local_cells[i*N + j];
        }
    }
}

template<int N>
__device__ void updateCandidatesLocal(uint32_t* shared_cells, int tid) {
    const int SIZE = N * N;
    
    // Allocate local memory for this thread
    uint32_t local_cells[N*N];
    //int status = 0;
    
    // Each thread processes one row, one column, and one box
    for (int iter = 0; iter < 3; iter++) {
        // Process rows
        loadGroup<N>(shared_cells, local_cells, tid*SIZE, 1, N);
        updateGrpCandidtLocal<N>(local_cells);
        removeNakedPairsLocal<N>(local_cells);
        __syncthreads();
        storeGroup<N>(shared_cells, local_cells, tid*SIZE, 1, N);
        __syncthreads();
        
        // if (invalid) {
        //     shared_cells[0] = 0;
        //     return;
        // }

        // Process columns
        loadGroup<N>(shared_cells, local_cells, tid, SIZE, SIZE*N);
        updateGrpCandidtLocal<N>(local_cells);
        removeNakedPairsLocal<N>(local_cells);
        __syncthreads();
        storeGroup<N>(shared_cells, local_cells, tid, SIZE, SIZE*N);
        __syncthreads();
        
        // if (invalid) {
        //     shared_cells[0] = 0;
        //     return;
        // }

        // Process boxes
        int box_row = (tid / N) * N;
        int box_col = (tid % N) * N;
        loadGroup<N>(shared_cells, local_cells, box_row*SIZE + box_col, 1, SIZE);
        updateGrpCandidtLocal<N>(local_cells);
        removeNakedPairsLocal<N>(local_cells);
        __syncthreads();
        storeGroup<N>(shared_cells, local_cells, box_row*SIZE + box_col, 1, SIZE);
        __syncthreads();
        
        // if (invalid) {
        //     shared_cells[0] = 0;
        //     return;
        // }
    }
}

template<int SIZE>
__device__  SudokuCandidates<SIZE> proposalToCandidates(const  SudokuProposal<SIZE>& prop){
     SudokuCandidates<SIZE> candidates;
    for (int i = 0; i < SIZE * SIZE; i++) {
        if (prop.cells[i] == 0) {
            candidates.cells[i] = (1 << SIZE) - 1;  // All candidates
        } else {
            candidates.cells[i] = (1 << (prop.cells[i] - 1)) | FIXED_BIT;  // Fixed value
        }
    }
    return candidates;
}

template<int SIZE>
__device__  SudokuProposal<SIZE> candidatesToProposal(const  SudokuCandidates<SIZE>& candidates) {
     SudokuProposal<SIZE> prop;
    for (int i = 0; i < SIZE * SIZE; i++) {
        if (candidates.cells[i] & FIXED_BIT) {
            prop.cells[i] = 32 - __clz(candidates.cells[i] & ~FIXED_BIT);
        } else {
            prop.cells[i] = 0;
        }
    }
    return prop;
}

template<int SIZE>
__device__ void proposalToCandidatesLocal(uint32_t *cand_cells, const char *prop_cells, int tid){
    for (int i = 0; i < SIZE; i++) {
        int idx = SIZE*tid + i;
        if (prop_cells[idx] == 0) {
            cand_cells[idx] = (1 << SIZE) - 1;  // All candidates
        } else {
            cand_cells[idx] = (1 << (prop_cells[idx] - 1)) | FIXED_BIT;  // Fixed value
        }
    }
}

template<int SIZE>
__device__ void candidatesToProposalLocal(char *prop_cells, const uint32_t *cand_cells,  int tid){
    for (int i = 0; i < SIZE; i++) {
        int idx = SIZE*tid + i;
        if (cand_cells[idx] & FIXED_BIT) {
            prop_cells[i] = 32 - __clz(cand_cells[idx] & ~FIXED_BIT);
        } else {
            prop_cells[i] = 0;
        }
    }
}

template<int N>
__global__ void generateProposals(
    SudokuProposal<N*N>* proposals,
    SudokuProposal<N*N>* output_proposals,
    int *valid_flags,
    int* proposal_count,
    int max_proposals,
    int *solution
) {
    const int SIZE = N * N;
    const int nbranch = SIZE;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *proposal_count) return;
    
     SudokuProposal<SIZE> currentProp = proposals[tid];
     SudokuCandidates<SIZE> current = proposalToCandidates<SIZE>(currentProp);
    
    // Update candidates
    updateCandidates<N>(current.cells);
    currentProp = candidatesToProposal<SIZE>(current);

    CellVote thread_vote = findLeastCandidates<SIZE>(current);
    
    for (int i = 0; i < nbranch; i++) {
        valid_flags[tid * nbranch + i] = 0;
    }   

    // check if invalid or solved
    if (thread_vote.cell_idx == -1) {
        //*finished = true;
        //return;

        // check if we found a solution
        if (isSolution<SIZE>(currentProp)) {
            // found a solution
            output_proposals[tid * nbranch + 1] = currentProp;
            
            valid_flags[tid * nbranch + 1] = 1;
            *solution = tid * nbranch + 1;
        }
        return;
    }

    int best_cell = thread_vote.cell_idx;
    
    uint32_t candidates = current.cells[best_cell];

    int count = 0;
    for (int bit = 0; bit < SIZE; bit++) {
        if (candidates & (1u << bit)) {
            currentProp.cells[best_cell] = bit + 1;
            valid_flags[tid * nbranch + count] = 1;
            output_proposals[tid * nbranch + count] = currentProp;
            count++;
        }
    }
}

template<int N, int M> // N: size of grid, M: number of grids per block (e.g. 3 for 9x9, 2 for 16x16)
__global__ void generateProposalsLocal(
    SudokuProposal<N*N>* proposals,
    SudokuProposal<N*N>* output_proposals,
    int *valid_flags,
    int* proposal_count,
    int max_proposals,
    int *solution
) {
    const int SIZE = N * N;
    const int nbranch = SIZE;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int bid = blockIdx.x * gridDim.y + blockIdx.y;

    int gid = tid / SIZE; // grid id
    int wid = tid % SIZE; // within grid id

    if (gid >= M) return;

    int pid = bid * M + gid; // proposal id

    if (pid >= *proposal_count) return;
    
    int out_pid = pid * nbranch;

    // Shared memory for this proposal's grid
    __shared__ uint32_t shared_cells[M*SIZE*SIZE];
    __shared__ int is_complete[M];
    __shared__ int shared_count[M];
    __shared__ int shared_idx[M];

    uint32_t *shared_cells_grid = shared_cells + gid*SIZE*SIZE;
    int *is_complete_grid = is_complete + gid;
    int *shared_count_grid = shared_count + gid;
    int *shared_idx_grid = shared_idx + gid;
    
    char prop_row[SIZE];

    // Convert proposal to candidates in shared memory
    proposalToCandidatesLocal<SIZE>(shared_cells_grid, proposals[pid].cells, wid);
    __syncthreads();

    // Update candidates
    updateCandidatesLocal<N>(shared_cells_grid, wid);
    __syncthreads();
    
    // Find best cell cooperatively
    findLeastCandidatesLocal<SIZE>(shared_cells_grid, wid, shared_count_grid, shared_idx_grid);
    __syncthreads();
    
    candidatesToProposalLocal<SIZE>(prop_row, shared_cells_grid, wid);

    // Early exit check - if invalid or solved    
    // -1: no candidate cells found (solved?)
    // -2: invalid proposal
    if (*shared_idx_grid == -2) {
        // Invalid proposal
        valid_flags[out_pid + wid] = 0;
        return;
    }
    if (*shared_idx_grid == -1) {
        if(true){
            char *out_cells = output_proposals[out_pid + 1].cells;
            for (int i = 0; i < SIZE; i++) {
                out_cells[SIZE*wid + i] = prop_row[i];
            }

            if(wid == 0){
                valid_flags[out_pid + 1] = 1;
                atomicExch(solution, out_pid + 1);
            }
        }
        return;
    }
    __syncthreads();
    
    // Each thread handles one potential value for the best cell
    int best_cell = *shared_idx_grid;
    uint32_t candidates = shared_cells_grid[best_cell];

    for (int bit = 0; bit < nbranch; bit++) {
        if (candidates & (1u << bit)) {
            char *out_cells = output_proposals[out_pid + bit].cells;
            for (int i = 0; i < SIZE; i++) {
                out_cells[SIZE*wid + i] = prop_row[i];
            }
            
            // Set the proposed value
            if (wid == 0) {
                output_proposals[out_pid + bit].cells[best_cell] = bit + 1;
                valid_flags[out_pid + bit] = 1;
            }
        } else {
            if (wid == 0) {
                valid_flags[out_pid + bit] = 0;
            }
        }
    }
}

// Compact proposals using scan results
template<int SIZE>
__global__ void compactifyProposals(
    const SudokuProposal<SIZE>* input_proposals,
    SudokuProposal<SIZE>* output_proposals,
    const int* valid_flags,
    const int* scan_results,
    int proposal_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= proposal_count) return;
    
    if (valid_flags[tid]) {
        int output_idx = scan_results[tid];
        output_proposals[output_idx] = input_proposals[tid];
    }
}

int popcount_cpu(uint32_t x) {
    return __builtin_popcount(x);
}

template <int N>
SudokuProposal<N*N> solveSudoku(std::vector<char>& grid) {    
    const int SIZE = N * N;

    SudokuProposal<SIZE> initial;
    // copy grid values to initial
    for (int i = 0; i < SIZE*SIZE; i++) {
        initial.cells[i] = grid[i];
    }
    
    // Allocate device memory
    SudokuProposal<SIZE>* d_proposals;
    SudokuProposal<SIZE>* d_output_proposals;
    SudokuProposal<SIZE>* d_saved_proposals;
    int *d_proposal_count;
    int *d_foundSolution;

    cudaMalloc(&d_proposals, MAX_PROPOSALS * sizeof(SudokuProposal<SIZE>));
    cudaMalloc(&d_output_proposals, MAX_PROPOSALS * sizeof(SudokuProposal<SIZE>));
    cudaMalloc(&d_saved_proposals, MAX_PROPOSALS * sizeof(SudokuProposal<SIZE>));
    cudaMalloc(&d_proposal_count, sizeof(int));
    cudaMalloc(&d_foundSolution, sizeof(int));

    // Copy initial proposal to device
    cudaMemcpy(d_proposals, &initial, sizeof(SudokuProposal<SIZE>), cudaMemcpyHostToDevice);
    int initial_count = 1;
    cudaMemcpy(d_proposal_count, &initial_count, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_foundSolution, new int(0), sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_vector<int> d_valid_flags(MAX_PROPOSALS);
    thrust::device_vector<int> d_scan_results(MAX_PROPOSALS);

    SudokuProposal<SIZE> result;

    const int max_branches = SIZE;
    int n_saved_proposals = 0;
    int total_proposals = 0;
    uint64_t kernel_time = 0;
    int n_iters = 0;

    // Main solving loop
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        int host_count;

        cudaMemcpy(&host_count, d_proposal_count, sizeof(int), cudaMemcpyDeviceToHost);
        //std::cout << "Current count: " << host_count << std::endl;

        if (host_count > MAX_THREADS) {
            // only explore the bottom MAX_THREADS proposals,
            // saving the rest for the next iteration.      
            n_saved_proposals = host_count - MAX_THREADS;
            host_count = MAX_THREADS;

            // Copy the proposals to be saved to d_saved_proposals
            cudaMemcpy(d_saved_proposals, d_proposals + host_count, n_saved_proposals * sizeof(SudokuProposal<SIZE>), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_proposal_count, &host_count, sizeof(int), cudaMemcpyHostToDevice);
        }

        total_proposals += host_count;

        int num_blocks = (host_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto k_start = std::chrono::high_resolution_clock::now();
        if (false){
            generateProposals<N><<<num_blocks, BLOCK_SIZE>>>(
                d_proposals,
                d_output_proposals,
                thrust::raw_pointer_cast(d_valid_flags.data()),
                d_proposal_count,
                MAX_PROPOSALS,
                d_foundSolution
            );
        } else {
            const int BLOCK_SIZE_X = 32;  // number of threads
            const int BLOCK_SIZE_Y = BLOCK_SIZE / BLOCK_SIZE_X; // number of threads

            const int GRIDS_PER_BLOCK_X = BLOCK_SIZE_X / SIZE;
            const int GRIDS_PER_BLOCK_Y = BLOCK_SIZE_Y;
            const int GRIDS_PER_BLOCK = GRIDS_PER_BLOCK_X * GRIDS_PER_BLOCK_Y;

            int n_blocks_y = (host_count + GRIDS_PER_BLOCK - 1) / GRIDS_PER_BLOCK;

            // print blocks and grids 
            generateProposalsLocal<N, GRIDS_PER_BLOCK><<<dim3(1, n_blocks_y), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(
                d_proposals,
                d_output_proposals,
                thrust::raw_pointer_cast(d_valid_flags.data()),
                d_proposal_count,
                MAX_PROPOSALS,
                d_foundSolution
            );
        }

        n_iters++;
        cudaDeviceSynchronize();
        auto k_end = std::chrono::high_resolution_clock::now();
        kernel_time += std::chrono::duration_cast<std::chrono::microseconds>(k_end - k_start).count();

        // check if we found a solution
        int foundSolution;
        cudaMemcpy(&foundSolution, d_foundSolution, sizeof(int), cudaMemcpyDeviceToHost);
        if (foundSolution) {
            // Extract the solution 
            cudaMemcpy(&result, d_output_proposals + foundSolution, sizeof(SudokuProposal<SIZE>), cudaMemcpyDeviceToHost);

            break;
        }
        
        host_count *= max_branches;  // Each proposal generates n new ones

        // Perform exclusive scan
        thrust::exclusive_scan(
            thrust::device,
            d_valid_flags.begin(),
            d_valid_flags.begin() + host_count,
            d_scan_results.begin()
        );

        // Compact the proposals
        num_blocks = (host_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compactifyProposals<SIZE><<<num_blocks, BLOCK_SIZE>>>(
            d_output_proposals,
            d_proposals,
            thrust::raw_pointer_cast(d_valid_flags.data()),
            thrust::raw_pointer_cast(d_scan_results.data()),
            host_count
        );

        // Get new proposal count
        host_count = d_scan_results[host_count-1] + d_valid_flags[host_count - 1];
        
        // Copy saved proposals back to d_proposals
        if (n_saved_proposals > 0) {
            cudaMemcpy(d_proposals + host_count, d_saved_proposals, n_saved_proposals * sizeof(SudokuProposal<SIZE>), cudaMemcpyDeviceToDevice);
            
            host_count += n_saved_proposals;
            n_saved_proposals = 0;
        }
        cudaMemcpy(d_proposal_count, &host_count, sizeof(int), cudaMemcpyHostToDevice);

        if (host_count == 0){
            printf("No more proposals\n");
            // return first proposal
            cudaMemcpy(&result, d_proposals, sizeof(SudokuProposal<SIZE>), cudaMemcpyDeviceToHost);
            break;
        }
        if (host_count >= MAX_PROPOSALS) {
            printf("Warning: Reached maximum proposal limit\n");
            result = initial;
            break;
        }
        
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    std::cout << "Proposal kernel time: " << kernel_time / 1e6 << " s" << std::endl;
    std::cout << "Iterations: " << n_iters << std::endl;
    std::cout << "Average kernel time: " << kernel_time / n_iters / 1e3 << " ms" << std::endl;
    std::cout << "Total proposals evaluated: " << total_proposals << std::endl;
    std::cout << "Proposals per second: " << total_proposals / elapsed.count() << std::endl;

    // Cleanup
    cudaFree(d_proposals);
    cudaFree(d_output_proposals);
    cudaFree(d_proposal_count);
    cudaFree(d_foundSolution);

    return result;
}

class SudokuParser {
private:
    // Helper function to check if a character is a valid Sudoku digit
    bool isValidDigit(char c, int maxDigit) {
        if (isdigit(c)) return c != '0';
        return c >= 'A' && c <= ('A' + maxDigit - 11); // For digits > 9
    }

    // Convert character to integer value
    int charToValue(char c) {
        if (isdigit(c)) return c - '0';
        if (c >= 'A' && c <= 'Z') return 10 + (c - 'A');
        return 0; // Return 0 for invalid or empty cells
    }

    // Determine grid size from input file
    int detectGridSize(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        std::getline(file, line);
        file.seekg(0);

        if (line.find('|') != std::string::npos) {
            // For formatted input, count numbers in first valid line
            while (std::getline(file, line)) {
                if (line.find('|') != std::string::npos) {
                    int count = 0;
                    for (char c : line) {
                        if (isdigit(c) || (c >= 'A' && c <= 'Z')) count++;
                    }
                    file.close();
                    return count;
                }
            }
        } else {
            // For simple input, find first line with numbers and count them
            while (std::getline(file, line)) {
                int count = 0;
                for (char c : line) {
                    if (isdigit(c) || (c >= 'A' && c <= 'Z')) count++;
                }
                if (count > 0) {
                    file.close();
                    return count;
                }
            }
        }

        file.close();
        throw std::runtime_error("Could not determine grid size");
    }

public:
    std::vector<std::vector<int>> parseFile(const std::string& filename) {
        int gridSize = detectGridSize(filename);
        
        // Verify grid size is a perfect square
        int boxSize = std::sqrt(gridSize);
        if (boxSize * boxSize != gridSize) {
            throw std::runtime_error("Invalid grid size: " + std::to_string(gridSize) + 
                                   " (must be a perfect square)");
        }

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::vector<std::vector<int>> grid(gridSize, std::vector<int>(gridSize, 0));
        std::string line;

        // Read first line to determine format
        std::getline(file, line);
        file.seekg(0);

        if (line.find('-') != std::string::npos || line.find('|') != std::string::npos) {
            parseFormattedInput(file, grid, gridSize);
        } else {
            parseSimpleInput(file, grid, gridSize);
        }

        file.close();
        return grid;
    }

private:
    void parseFormattedInput(std::ifstream& file, std::vector<std::vector<int>>& grid, int gridSize) {
        std::string line;
        int row = 0;

        while (std::getline(file, line) && row < gridSize) {
            if (line.find('|') == std::string::npos) continue; // Skip separator lines

            int col = 0;
            for (char c : line) {
                if (isdigit(c) || (c >= 'A' && c <= 'Z')) {
                    if (col < gridSize) {
                        grid[row][col++] = charToValue(c);
                    }
                }
            }
            if (col > 0) row++; // Only increment row if we found numbers
        }
    }

    void parseSimpleInput(std::ifstream& file, std::vector<std::vector<int>>& grid, int gridSize) {
        std::string line;
        int row = 0, col = 0;

        while (std::getline(file, line) && row < gridSize) {
            for (char c : line) {
                if (isdigit(c) || (c >= 'A' && c <= 'Z')) {
                    grid[row][col++] = charToValue(c);
                    if (col == gridSize) {
                        col = 0;
                        row++;
                    }
                }
            }
        }
    }
};


void printSudoku(char grid[], int boxSize) {
    int gridSize = boxSize * boxSize;

    int nw = gridSize > 9 ? 3 : 2;
    int ndash = gridSize*nw + (boxSize+1)*2 - 1;
    // Print the parsed grid
    for (int i = 0; i < gridSize; i++) {
        if (i % boxSize == 0) {
            std::cout << std::string(ndash, '-') << std::endl;
        }
        for (int j = 0; j < gridSize; j++) {
            if (j % boxSize == 0) std::cout << "| ";
            char v = grid[i * gridSize + j];
            if (v == 0) {
                if (gridSize <= 9) {
                    std::cout << "  ";
                } else {
                    std::cout << "   ";
                }
            } else {
                if (gridSize <= 9) {
                    printf("%1d ", v);
                } else {
                    printf("%2d ", v);
                }
            }
        }
        std::cout << "|" << std::endl;
    }
    std::cout << std::string(ndash, '-') << std::endl;
}

void printSudokuColor(const char unsolved[], const char solved[], int boxSize) {
    int gridSize = boxSize * boxSize;

    // ANSI color codes
    const char* BLUE = "\033[34m";
    const char* RESET = "\033[0m";

    int nw = gridSize > 9 ? 3 : 2;
    int ndash = gridSize*nw + (boxSize+1)*2 - 1;
    
    for (int i = 0; i < gridSize; i++) {
        if (i % boxSize == 0) {
            std::cout << std::string(ndash, '-') << std::endl;
        }
        for (int j = 0; j < gridSize; j++) {
            if (j % boxSize == 0) std::cout << "| ";
            
            int idx = i * gridSize + j;
            char v = solved[idx];
            bool was_unsolved = (unsolved[idx] == 0);
            
            if (v == 0) {
                if (gridSize <= 9) {
                    std::cout << "  ";
                } else {
                    std::cout << "   ";
                }
            } else {
                if (was_unsolved) std::cout << BLUE;
                if (gridSize <= 9) {
                    printf("%1d ", v);
                } else {
                    printf("%2d ", v);
                }
                if (was_unsolved) std::cout << RESET;
            }
        }
        std::cout << "|" << std::endl;
    }
    std::cout << std::string(ndash, '-') << std::endl;
}

// Example usage
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <sudoku_file>" << std::endl;
        return 1;
    }

    try {
        SudokuParser parser;
        auto grid = parser.parseFile(argv[1]);
        int gridSize = grid.size();
        int boxSize = std::sqrt(gridSize);

        // flatten grid 
        std::vector<char> flat_grid;
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                flat_grid.push_back(grid[i][j]);
            }
        }

        // Print the parsed grid
        printSudoku(flat_grid.data(), boxSize);

        // Solve the Sudoku
        //SudokuProposal<SIZE> solution;
        char *cells;
        if (boxSize == 3) {
            SudokuProposal<9> solution = solveSudoku<3>(flat_grid);
            cells = solution.cells;
        } else if (boxSize == 4) {
            SudokuProposal<16> solution = solveSudoku<4>(flat_grid);
            cells = solution.cells;
        } else if (boxSize == 5) {
            SudokuProposal<25 >solution = solveSudoku<5>(flat_grid);
            cells = solution.cells;
        } else {
            throw std::runtime_error("Unsupported grid size: " + std::to_string(gridSize));
        }

        // Print the solution
        std::cout << "Solution:" << std::endl;
        printSudokuColor(flat_grid.data(), cells, boxSize);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}