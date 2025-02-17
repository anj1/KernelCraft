#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
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
const int MAX_SIZE = 25;
const int MAX_POSSIBILITIES = 32; // Using 32-bit masks
const int MAX_THREADS = 5 * 1024;
const int MAX_PROPOSALS = 512 * 1024; // Maximum number of proposals in global memory
const int BLOCK_SIZE = 32;

#define FIXED_BIT (1u << 31)

struct SudokuCandidates {
    uint32_t cells[MAX_SIZE * MAX_SIZE];  // Possibility masks for each cell
};
struct SudokuProposal {
    char cells[MAX_SIZE * MAX_SIZE];  // Fixed values 
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
__device__ CellVote findLeastCandidates(const SudokuCandidates& prop) {
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
__device__ bool isSolution(const SudokuProposal& prop) {
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

template<int SIZE>
__device__ SudokuCandidates proposalToCandidates(const SudokuProposal& prop){
    SudokuCandidates candidates;
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
__device__ SudokuProposal candidatesToProposal(const SudokuCandidates& candidates) {
    SudokuProposal prop;
    for (int i = 0; i < SIZE * SIZE; i++) {
        if (candidates.cells[i] & FIXED_BIT) {
            prop.cells[i] = 32 - __clz(candidates.cells[i] & ~FIXED_BIT);
        } else {
            prop.cells[i] = 0;
        }
    }
    return prop;
}

template<int N>
__global__ void generateProposals(
    SudokuProposal* proposals,
    SudokuProposal* output_proposals,
    int *valid_flags,
    int* proposal_count,
    int max_proposals,
    int *solution
) {
    const int SIZE = N * N;
    const int nbranch = SIZE;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *proposal_count) return;
    
    SudokuProposal currentProp = proposals[tid];
    SudokuCandidates current = proposalToCandidates<SIZE>(currentProp);
    
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

// Compact proposals using scan results
__global__ void compactifyProposals(
    const SudokuProposal* input_proposals,
    SudokuProposal* output_proposals,
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
SudokuProposal solveSudoku(std::vector<char>& grid) {    
    const int SIZE = N * N;

    SudokuProposal initial;
    // copy grid values to initial
    for (int i = 0; i < SIZE*SIZE; i++) {
        initial.cells[i] = grid[i];
    }
    
    // Allocate device memory
    SudokuProposal* d_proposals;
    SudokuProposal* d_output_proposals;
    SudokuProposal* d_saved_proposals;
    int *d_proposal_count;
    int *d_foundSolution;

    cudaMalloc(&d_proposals, MAX_PROPOSALS * sizeof(SudokuProposal));
    cudaMalloc(&d_output_proposals, MAX_PROPOSALS * sizeof(SudokuProposal));
    cudaMalloc(&d_saved_proposals, MAX_PROPOSALS * sizeof(SudokuProposal));
    cudaMalloc(&d_proposal_count, sizeof(int));
    cudaMalloc(&d_foundSolution, sizeof(int));

    // Copy initial proposal to device
    cudaMemcpy(d_proposals, &initial, sizeof(SudokuProposal), cudaMemcpyHostToDevice);
    int initial_count = 1;
    cudaMemcpy(d_proposal_count, &initial_count, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_foundSolution, new int(0), sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_vector<int> d_valid_flags(MAX_PROPOSALS);
    thrust::device_vector<int> d_scan_results(MAX_PROPOSALS);

    SudokuProposal result;

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
            cudaMemcpy(d_saved_proposals, d_proposals + host_count, n_saved_proposals * sizeof(SudokuProposal), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_proposal_count, &host_count, sizeof(int), cudaMemcpyHostToDevice);
        }

        total_proposals += host_count;

        int num_blocks = (host_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto k_start = std::chrono::high_resolution_clock::now();
        generateProposals<N><<<num_blocks, BLOCK_SIZE>>>(
            d_proposals,
            d_output_proposals,
            thrust::raw_pointer_cast(d_valid_flags.data()),
            d_proposal_count,
            MAX_PROPOSALS,
            d_foundSolution
        );
        n_iters++;
        cudaDeviceSynchronize();
        auto k_end = std::chrono::high_resolution_clock::now();
        kernel_time += std::chrono::duration_cast<std::chrono::microseconds>(k_end - k_start).count();

        // check if we found a solution
        int foundSolution;
        cudaMemcpy(&foundSolution, d_foundSolution, sizeof(int), cudaMemcpyDeviceToHost);
        if (foundSolution) {
            // Extract the solution 
            cudaMemcpy(&result, d_output_proposals + foundSolution, sizeof(SudokuProposal), cudaMemcpyDeviceToHost);

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
        compactifyProposals<<<num_blocks, BLOCK_SIZE>>>(
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
            cudaMemcpy(d_proposals + host_count, d_saved_proposals, n_saved_proposals * sizeof(SudokuProposal), cudaMemcpyDeviceToDevice);
            
            host_count += n_saved_proposals;
            n_saved_proposals = 0;
        }
        cudaMemcpy(d_proposal_count, &host_count, sizeof(int), cudaMemcpyHostToDevice);

        if (host_count == 0){
            printf("No more proposals\n");
            // return first proposal
            cudaMemcpy(&result, d_proposals, sizeof(SudokuProposal), cudaMemcpyDeviceToHost);
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
        SudokuProposal solution;
        if (boxSize == 3) {
            solution = solveSudoku<3>(flat_grid);
        } else if (boxSize == 4) {
            solution = solveSudoku<4>(flat_grid);
        } else if (boxSize == 5) {
            solution = solveSudoku<5>(flat_grid);
        } else {
            throw std::runtime_error("Unsupported grid size: " + std::to_string(gridSize));
        }

        // Print the solution
        std::cout << "Solution:" << std::endl;
        printSudokuColor(flat_grid.data(), solution.cells, boxSize);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}