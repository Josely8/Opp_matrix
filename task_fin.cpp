#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <cmath>
#include <mpi.h>
#include <omp.h>

// разкоментить соответвующий способ
#define MODE_OMP          // OpenMP
// #define MODE_MPI         // MPI
// #define MODE_MPI_OMP     // MPI + OpenMP

class Matrix {
    std::vector<double> data;
    int rows, cols;
public:
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    
    double& operator()(int i, int j) { 
        return data[i * cols + j]; 
    }
    
    const double& operator()(int i, int j) const { 
        return data[i * cols + j]; 
    }
    
    void fillRandom() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                (*this)(i, j) = dis(gen);
            }
        }
    }

    void fillIdentical() {
        for (int i = 0; i < std::min(rows, cols); ++i) {
            (*this)(i, i) = 1.0;
        }
    }
    
    void print(int limit = 5) const {
        int r = std::min(rows, limit);
        int c = std::min(cols, limit);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                std:: cout << (*this)(i, j) << " ";
            }
            std::cout << (c < cols ? " ..." : "") << "\n";
        }
        if (r < rows) std::cout << "...\n";
    }
    
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    double* getData() { return data.data(); }
    const double* getData() const { return data.data(); }
};


class SerialMultiplier {
    int M, N, K;
public:
    SerialMultiplier(int m, int n, int k) : M(m), N(n), K(k) {}
    
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double el = 0.0;
                for (int k = 0; k < K; k++) {
                    el += A(i, k) * B(k, j);
                }
                C(i, j) = el;
            }
        }
    }
};


// OpenMP
class OMPMultiplier {
    int M, N, K;
    int num_threads;
    
public:
    OMPMultiplier(int m, int n, int k, int threads = 1) : M(m), N(n), K(k), num_threads(threads) {
        omp_set_num_threads(num_threads);
        std::cout << "OpenMP, кол-во процессов = " << num_threads << std::endl;
    }
    
    void multiply(const Matrix& A, const Matrix& B, Matrix& C) {
        double total_time = 0;

        #pragma omp parallel reduction(+:total_time)
        {

            double start_time, end_time, total_t_time;
            int id = omp_get_thread_num();
            start_time = omp_get_wtime();

            #pragma omp for collapse(2)
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    double el = 0.0;
                    for (int k = 0; k < K; k++) {
                        el += A(i, k) * B(k, j);
                    }
                    C(i, j) = el;
                }
            }
            end_time = omp_get_wtime();
            total_t_time = end_time - start_time;
            std::cout << total_t_time << " ";
            total_time += total_t_time;
        }
        std::cout << "\n================\n" << total_time << "\n================" << std::endl;
        
        
    }
};

// MPI
class MPIMultiplier {
    int rank, size;
    int M, N, K;
    
public:
    MPIMultiplier(int m, int n, int k) : M(m), N(n),  K(k),rank(0), size(1) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (rank == 0) {
            std::cout << "MPI, кол-во процессов = " << size << std::endl;
        }
    }
    
    void multiply(const Matrix& A, const Matrix& B, Matrix& C, bool is_root = true) {
        if (!is_root) {
            int dim[3];
            MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
            M = dim[0];
            N = dim[1];
            K = dim[2];
        } else {
            int dim[3] = {M, N, K};
            MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        int procces_rows = M / size;
        int remaining_rows = M % size;
        if (rank < remaining_rows) {
            procces_rows++;
        }

        Matrix local_A(procces_rows, K);
        Matrix local_C(procces_rows, N);
        
        std::vector<int> send_counts(size, 0);
        std::vector<int> offsets(size, 0);
        
        if (is_root) {
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int rows_for_proc = M / size;
                if (i < remaining_rows) {
                    rows_for_proc++;
                }
                send_counts[i] = rows_for_proc * K;
                offsets[i] = offset;
                offset += send_counts[i];
            }
        }
        
        MPI_Scatterv(is_root ? A.getData() : nullptr, 
                    send_counts.data(), offsets.data(), MPI_DOUBLE,
                    local_A.getData(), procces_rows * K, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        Matrix B_copy(K, N);
        if (is_root) {
            B_copy = B;
        }        
        MPI_Bcast(B_copy.getData(), K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        for (int i = 0; i < procces_rows; i++) {
            for (int j = 0; j < N; j++) {
                double el = 0.0;
                for (int k = 0; k < K; k++) {
                    el += local_A(i, k) * B(k, j);
                }
                local_C(i, j) = el;
            }
        }
        
        std::vector<int> gather_counts(size);
        std::vector<int> gather_offsets(size);
        
        if (is_root) {
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int rows_for_proc = send_counts[i] / K;
                gather_counts[i] = rows_for_proc * N;
                gather_offsets[i] = offset;
                offset += gather_counts[i];
            }
        }
        
        MPI_Gatherv(local_C.getData(), procces_rows * N, MPI_DOUBLE,
                   is_root ? C.getData() : nullptr, 
                   gather_counts.data(), gather_offsets.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }
    
    bool isRoot() const { return rank == 0; }
    int getRank() const { return rank; }
    int getSize() const { return size; }
};


// MPI+OpenMP
class HybridMultiplier {
    int rank, size;
    int M, K, N;
    int num_threads;
    
public:
    HybridMultiplier(int m, int n, int k, int threads = 1) : M(m), N(n), K(k), num_threads(threads) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        omp_set_num_threads(num_threads);
        if (rank == 0) {
            std::cout << "MPI+OpenMP, кол-во процессов MPI = " << size << "<...> OpenMP = " << num_threads << std::endl;
        }
    }
    
    void multiply(const Matrix& A, const Matrix& B, Matrix& C, bool is_root = true) {
        if (!is_root) {
            int dim[3];
            MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
            M = dim[0];
            N = dim[1];
            K = dim[2];
        } else {
            int dim[3] = {M, N, K};
            MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
        }

        int procces_rows = M / size;
        int remaining_rows = M % size;
        if (rank < remaining_rows) {
            procces_rows++;
        }

        Matrix local_A(procces_rows, K);
        Matrix local_C(procces_rows, N);

        std::vector<int> send_counts(size);
        std::vector<int> offsets(size);
        
        if (is_root) {
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int rows_for_proc = M / size;
                if (i < remaining_rows) {
                    rows_for_proc++;
                }
                send_counts[i] = rows_for_proc * K;
                offsets[i] = offset;
                offset += send_counts[i];
            }
        }

        MPI_Scatterv(is_root ? A.getData() : nullptr, 
                    send_counts.data(), offsets.data(), MPI_DOUBLE,
                    local_A.getData(), procces_rows * K, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        double total_time = 0;

        #pragma omp parallel reduction(+:total_time)
        {

            double start_time, end_time, total_t_time;
            start_time = omp_get_wtime();

            int i, j, k;

            #pragma omp for collapse(2)
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    double el = 0.0;
                    for (k = 0; k < K; k++) {
                        el += A(i, k) * B(k, j);
                    }
                    local_C(i, j) = el;
                }
            }
            end_time = omp_get_wtime();
            total_t_time = end_time - start_time;
            total_time += total_t_time;
        }
        std::cout << "\n================\n" << total_time << "\n================" << std::endl;

        std::vector<int> gather_counts(size);
        std::vector<int> gather_offsets(size);
        
        if (is_root) {
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int rows_for_proc = send_counts[i] / K;
                gather_counts[i] = rows_for_proc * N;
                gather_offsets[i] = offset;
                offset += gather_counts[i];
            }
        }
        
        MPI_Gatherv(local_C.getData(), procces_rows * N, MPI_DOUBLE,
                   is_root ? C.getData() : nullptr, 
                   gather_counts.data(), gather_offsets.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }
    bool isRoot() const { return rank == 0; }
    int getRank() const { return rank; }
    int getSize() const { return size; }
};

double getTime() {
    #if defined(MODE_OMP)
        return omp_get_wtime();
    #elif defined(MODE_MPI) || defined(MODE_MPI_OMP)
        return MPI_Wtime();
    #else
        return static_cast<double>(clock()) / CLOCKS_PER_SEC;
    #endif
}


int main(int argc, char** argv) {
    #if defined(MODE_MPI) || defined(MODE_MPI_OMP)
        MPI_Init(&argc, &argv);
    #endif

    bool is_root = true;
    int rank = 0;
    int size = 1;
    #if defined(MODE_MPI) || defined(MODE_MPI_OMP)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        is_root = (rank == 0);
    #endif

    int M = 500;
    int N = 500;
    int K = 500;
    int omp_threads = 1;
    bool is_identical = false;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[3]);
        K = std::atoi(argv[2]);
    }
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-t") {
            if (i + 1 < argc) {
                omp_threads = std::atoi(argv[i + 1]);
                i++;
            }
        } else if (arg == "-I") {
            is_identical = true;
        }
    }
    
    if (is_root) {
        std::cout << "Размеры матриц: \nM = " << M << ", N = " << N << ", K = " << K << std::endl;
    }
    
    double start_time = 0, end_time = 0;
    double T1 = 0;
    double T0 = 0;
    Matrix *A = nullptr, *B = nullptr, *C = nullptr;
    
    if (is_root) {
        A = new Matrix(M, K);
        B = new Matrix(K, N);
        C = new Matrix(M, N);

        if (is_identical) {
            std::cout << "единичные матрицы"<< std::endl;
            A->fillIdentical();
            B->fillIdentical();
        } else {
            std::cout << "матрицы из случайных чисел"<< std::endl;
            srand(time(nullptr));
            A->fillRandom();
            B->fillRandom();
        }
        
        std::cout << "Матрица A(M x K):"<< std::endl;
        A->print();
        std::cout << "Матрица B(K x N):"<< std::endl;
        B->print();
    }

    if (is_root) {
        std::cout << "Однопоточное умножение"<< std::endl;
        SerialMultiplier multiplier(M, N, K);
        start_time = getTime();
        multiplier.multiply(*A, *B, *C);
        end_time = getTime();
        T0 = end_time - start_time;

        std::cout << "Время умножения(T0) = " << T0 << std::endl;
        std::cout << "Результат:"<< std::endl;
        C->print();
    }
        
    #if defined(MODE_OMP)
        if (is_root) {
            OMPMultiplier multiplier(M, K, N, omp_threads);
            std::cout << "OpenMP умножение"<< std::endl;
            start_time = getTime();
            multiplier.multiply(*A, *B, *C);
            end_time = getTime();
        }
        
    #elif defined(MODE_MPI)
        MPIMultiplier multiplier(M, N, K);
        
        if (is_root) {
            std::cout << "MPI умножение"<< std::endl;
            start_time = getTime();
        }
        
        if (is_root) {
            multiplier.multiply(*A, *B, *C, true);
        } else {
            Matrix A_1(1, 1);
            Matrix B_1(1, 1);
            Matrix C_1(1, 1);
            multiplier.multiply(A_1, B_1, C_1, false);
        }
        
        if (is_root) {
            end_time = getTime();
        }
        
    #elif defined(MODE_MPI_OMP)
        HybridMultiplier multiplier(M, N, K, omp_threads);
        
        if (is_root) {
            std::cout << "MPI+OMP умножение"<< std::endl;
            start_time = getTime();
        }
        
        if (is_root) {
            multiplier.multiply(*A, *B, *C, true);
        } else {
            Matrix A_1(1, 1);
            Matrix B_1(1, 1);
            Matrix C_1(1, 1);
            multiplier.multiply(A_1, B_1, C_1, false);
        }
        
        if (is_root) {
            end_time = getTime();
        }
    #endif
    
    if (is_root) {
        T1 = end_time - start_time;
        
        std::cout << "\nРезультат:" << std::endl;
        std::cout << "Время умножения(T1) = " << T1 << " секунд" << std::endl;
        std::cout << "Ускорение:" << T0 / T1 << std::endl;
        
        
        std::cout << "Результат умножения:" << std::endl;
        C->print();       

        delete A;
        delete B;
        delete C;
    }

    #if defined(MODE_MPI) || defined(MODE_MPI_OMP)
        MPI_Finalize();
    #endif
    return 0;
}