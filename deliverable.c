#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace std::chrono;

// Function to determine optimal block size based on matrix size
int getOptimalBlockSize(int n) {
	if (n <= 16) {
		return 4;
	} else if (n <= 128){
		return 8;
	} else if (n <= 1024) {
        return 32;
    } else if (n <= 4096) {
        return 64;
    } else {
		// For bigger matrices, not used in this case since the max is 4096x4096
        return 128;
    }
}

// Function to initialize a random n x n matrix
void initializeMatrix(vector<vector<float>> &matrix, int n) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// Function to inizitalize the matrix as symmetric
void makeSymmetric(vector<vector<float>> &matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            matrix[i][j] = matrix[j][i];
        }
    }
}

// Sequential symmetry check
bool checkSym(const vector<vector<float>> &matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (matrix[i][j] != matrix[j][i]) {
                return false;
            }
        }
    }
    return true;
}

// Sequential transpose function
void matTranspose(const vector<vector<float>> &matrix, vector<vector<float>> &transpose, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transpose[j][i] = matrix[i][j];
        }
    }
}

// Sequential block-based transpose
void matTransposeBlock(const vector<vector<float>>& matrix, vector<vector<float>>& transpose, int n, int block_size) {
    // Iterate over blocks
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            // Transpose within each block
            for (int bi = 0; bi < block_size && (i + bi) < n; ++bi) {
                for (int bj = 0; bj < block_size && (j + bj) < n; ++bj) {
                    transpose[j + bj][i + bi] = matrix[i + bi][j + bj];
                }
            }
        }
    }
}

// Parallel symmetry check using MPI
bool checkSymMPI(const vector<vector<float>>& matrix, int n, int rank, int size) {
    bool is_symmetric = true;
    int rows_per_proc = n / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? n : start_row + rows_per_proc;

    for (int i = start_row; i < end_row; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (abs(matrix[i][j] - matrix[j][i]) > 1e-6) {
                is_symmetric = false;
                break;
            }
        }
        if (!is_symmetric) break;
    }

    bool global_is_symmetric;
    MPI_Allreduce(&is_symmetric, &global_is_symmetric, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    return global_is_symmetric;
}

// Parallel transpose function using MPI
void matTransposeMPI(const vector<vector<float>>& matrix, vector<vector<float>>& transpose, int n, int rank, int size) {
    int rows_per_proc = n / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? n : start_row + rows_per_proc;

    vector<float> send_buf(n * rows_per_proc);
    vector<float> recv_buf(n * rows_per_proc);

    // Pack data into send buffer
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < n; ++j) {
            send_buf[i * n + j] = matrix[start_row + i][j];
        }
    }

    // Perform all-to-all communication
    MPI_Alltoall(send_buf.data(), rows_per_proc * rows_per_proc, MPI_FLOAT,
                 recv_buf.data(), rows_per_proc * rows_per_proc, MPI_FLOAT,
                 MPI_COMM_WORLD);

    // Unpack and transpose
    for (int p = 0; p < size; ++p) {
        for (int i = 0; i < rows_per_proc; ++i) {
            for (int j = 0; j < rows_per_proc; ++j) {
                transpose[start_row + j][p * rows_per_proc + i] = 
                    recv_buf[p * rows_per_proc * rows_per_proc + i * rows_per_proc + j];
            }
        }
    }
}

// Parallel block-based transpose using MPI
void matTransposeBlockMPI(const vector<vector<float>>& matrix,
                          vector<vector<float>>& transpose,
                          int n,
                          int rank,
                          int size,
                          int block_size)
{
    //----------------------------------------------------------------------
    // 0. Error checks
    //----------------------------------------------------------------------
    // Check that 'size' is a perfect square
    int sqrtP = static_cast<int>(std::sqrt(size));
	//Theoretically the ifs could be moved somewhere else to increase performance, but from tests the performance gained is negligible
    if (sqrtP * sqrtP != size) {
        if (rank == 0) {
            cerr << "[Error] Number of processes (" << size << ") is not a perfect square." 
                 << " A 2D decomposition requires sqrt(size)*sqrt(size) = size.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Check that 'n' is divisible by 'block_size'
    if (n % block_size != 0) {
        if (rank == 0) {
            cerr << "[Error] Matrix dimension n=" << n 
                 << " is not divisible by block_size=" << block_size << ".\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int blocks_per_dim = n / block_size;

    // Check that blocks_per_dim is at least as large as sqrtP 
    // (so that the assignment of blocks to processes is meaningful).
    if (blocks_per_dim < sqrtP || (blocks_per_dim % sqrtP) != 0) {
        if (rank == 0) {
            cerr << "[Error] The number of blocks per dimension (" << blocks_per_dim 
                 << ") must be >= sqrt(size)=" << sqrtP 
                 << " and also divisible by sqrtP to have a proper 2D block distribution.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    //----------------------------------------------------------------------
    // 1. Identify the 2D coordinates of this rank in the sqrtP x sqrtP grid
    //----------------------------------------------------------------------
    int procRow = rank / sqrtP;
    int procCol = rank % sqrtP;

    // Each process will handle blocks_per_proc blocks along each dimension
    int blocks_per_proc = blocks_per_dim / sqrtP;

    //----------------------------------------------------------------------
    // 2. Helper lambdas to find block ownership
    //----------------------------------------------------------------------
    auto ownerOfBlockOriginal = [&](int bi, int bj) -> int {
        // block (bi, bj) belongs to:
        //   rowOfProc = bi / blocks_per_proc
        //   colOfProc = bj / blocks_per_proc
        int ownerRow = bi / blocks_per_proc;
        int ownerCol = bj / blocks_per_proc;
        return ownerRow * sqrtP + ownerCol;
    };

    auto ownerOfBlockTransposed = [&](int bi, int bj) -> int {
        // block (bi, bj) in transpose corresponds to block (bj, bi) in original
        // So the owner is the owner of block (bj, bi) in the original
        int ownerRow = bj / blocks_per_proc;
        int ownerCol = bi / blocks_per_proc;
        return ownerRow * sqrtP + ownerCol;
    };

    //----------------------------------------------------------------------
    // 3. Actual block-by-block communication / local transpose
    //----------------------------------------------------------------------
    vector<float> blockBuf(block_size * block_size, 0.0f);

    for (int bi = 0; bi < blocks_per_dim; ++bi) {
        for (int bj = 0; bj < blocks_per_dim; ++bj) {

            int ownerOriginal   = ownerOfBlockOriginal(bi, bj);
            int ownerTransposed = ownerOfBlockTransposed(bi, bj);

            // Case A: I own the block (bi, bj) in the ORIGINAL matrix
            if (ownerOriginal == rank) {
                // 1) Copy the sub-block from 'matrix' to blockBuf
                int rowOffset = bi * block_size;
                int colOffset = bj * block_size;
                for (int r = 0; r < block_size; ++r) {
                    for (int c = 0; c < block_size; ++c) {
                        blockBuf[r * block_size + c] = matrix[rowOffset + r][colOffset + c];
                    }
                }

                // 2) If I also own the transposed block => local transpose
                if (ownerTransposed == rank) {
                    int tRowOffset = bj * block_size; // Because transpose block is (bj, bi)
                    int tColOffset = bi * block_size;
                    for (int r = 0; r < block_size; ++r) {
                        for (int c = 0; c < block_size; ++c) {
                            transpose[tRowOffset + c][tColOffset + r] = 
                                blockBuf[r * block_size + c];
                        }
                    }
                }
                else {
                    // Otherwise, send blockBuf to the owner of the transposed block
                    MPI_Send(blockBuf.data(),
                             block_size * block_size,
                             MPI_FLOAT,
                             ownerTransposed,
                             /*tag=*/0,
                             MPI_COMM_WORLD);
                }
            }

            // Case B: I own (bi, bj) in the TRANSPOSED matrix, but NOT in the original
            if (ownerTransposed == rank && ownerOriginal != rank) {
                // 1) Receive the block data from whoever owns (bi, bj) in the original
                MPI_Recv(blockBuf.data(),
                         block_size * block_size,
                         MPI_FLOAT,
                         ownerOriginal,
                         /*tag=*/0,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                // 2) Transpose the received block into my portion of 'transpose'
                int tRowOffset = bj * block_size;
                int tColOffset = bi * block_size;
                for (int r = 0; r < block_size; ++r) {
                    for (int c = 0; c < block_size; ++c) {
                        transpose[tRowOffset + c][tColOffset + r] = 
                            blockBuf[r * block_size + c];
                    }
                }
            }
        }
    }

    // Synchronize all processes before completing
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Check if matrix size was put
    if (argc != 2) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    int block_size = getOptimalBlockSize(n);

	// Check if matrix positive power of 2
    if (n <= 0 || (n & (n - 1)) != 0) {
        if (rank == 0) {
            cerr << "Matrix size must be a positive power of 2" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
	// Check if matrix divisible by number of processors
    if (n % size != 0) {
        if (rank == 0) {
            cerr << "Matrix size must be divisible by the number of processes" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        cout << "Using block size: " << block_size << endl;
    }

    vector<vector<float>> M(n, vector<float>(n));
    vector<vector<float>> T(n, vector<float>(n));

    // Initialize matrix only on rank 0
    if (rank == 0) {
        initializeMatrix(M, n);
		// To make symmetric use instead
		//makeSymmetric(M, n); 
        cout << "\n=== Sequential Implementation ===\n";
    }

    // Broadcast matrix to all processes
    for (int i = 0; i < n; ++i) {
        MPI_Bcast(M[i].data(), n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // Sequential operations
    if (rank == 0) {
        // Symmetry check
        auto startSymCheck = high_resolution_clock::now();
        bool isSymmetric = checkSym(M, n);
        auto endSymCheck = high_resolution_clock::now();
        auto durationSymCheck = duration_cast<microseconds>(endSymCheck - startSymCheck);

        cout << "Symmetry Check Time: " << durationSymCheck.count() << " microseconds\n";
        cout << "Matrix is" << (isSymmetric ? " " : " not ") << "symmetric\n";

        // Regular transpose
        auto startTranspose = high_resolution_clock::now();
        matTranspose(M, T, n);
        auto endTranspose = high_resolution_clock::now();
        auto durationTranspose = duration_cast<microseconds>(endTranspose - startTranspose);

        cout << "Regular Transpose Time: " << durationTranspose.count() << " microseconds\n";

        // Block transpose
        auto startBlockTranspose = high_resolution_clock::now();
        matTransposeBlock(M, T, n, block_size);
        auto endBlockTranspose = high_resolution_clock::now();
        auto durationBlockTranspose = duration_cast<microseconds>(endBlockTranspose - startBlockTranspose);

        cout << "Block Transpose Time: " << durationBlockTranspose.count() << " microseconds\n";
    }

    // Clear transpose matrix before parallel operations
    fill(T.begin(), T.end(), vector<float>(n, 0.0f));

    if (rank == 0) {
        cout << "\n=== Parallel Implementation ===\n";
    }

    // Parallel symmetry check
    auto startSymCheckMPI = high_resolution_clock::now();
    bool isSymmetricMPI = checkSymMPI(M, n, rank, size);
    auto endSymCheckMPI = high_resolution_clock::now();
    auto durationSymCheckMPI = duration_cast<microseconds>(endSymCheckMPI - startSymCheckMPI);

    if (rank == 0) {
        cout << "Symmetry Check Time: " << durationSymCheckMPI.count() << " microseconds\n";
        cout << "Matrix is" << (isSymmetricMPI ? " " : " not ") << "symmetric\n";
    }

    // Regular parallel transpose
    auto startTransposeMPI = high_resolution_clock::now();
    matTransposeMPI(M, T, n, rank, size);
    auto endTransposeMPI = high_resolution_clock::now();
    auto durationTransposeMPI = duration_cast<microseconds>(endTransposeMPI - startTransposeMPI);

    if (rank == 0) {
        cout << "Regular Transpose Time: " << durationTransposeMPI.count() << " microseconds\n";
    }

    // Block-based parallel transpose
    auto startBlockTransposeMPI = high_resolution_clock::now();
    matTransposeBlockMPI(M, T, n, rank, size, block_size);
    auto endBlockTransposeMPI = high_resolution_clock::now();
    auto durationBlockTransposeMPI = duration_cast<microseconds>(endBlockTransposeMPI - startBlockTransposeMPI);

    if (rank == 0) {
        cout << "Block Transpose Time: " << durationBlockTransposeMPI.count() << " microseconds\n";
    }

    MPI_Finalize();
    return 0;
}