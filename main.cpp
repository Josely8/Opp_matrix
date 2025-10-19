#include <iostream>
#include <vector>
#include <omp.h>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdlib>

class Matrix {
    std::vector<std::vector<double>> data_;
    int n_rows_, n_cols_;
public:
    explicit Matrix(int n_rows=0, int n_cols=0): n_rows_(n_rows), n_cols_(n_cols) {
        data_ = std::vector<std::vector<double>>(n_rows, std::vector<double>(n_cols, 0.0));
    }
    double& operator()(int i, int j);
    Matrix operator*(Matrix& other);
    Matrix mult_par(Matrix& other, int P);
    void fill_rand();
};

double& Matrix::operator()(int i, int j) {
    return data_[i][j];
}

Matrix Matrix::operator*(Matrix& other) {
    Matrix result(n_rows_, other.n_cols_);
    for (int i = 0; i < n_rows_; ++i) {
        for (int j = 0; j < other.n_cols_; ++j) {
            double el = 0;
            for (int k = 0; k < n_cols_; ++k) {
                el += (*this)(i, k) * other(k, j);
            }
            result(i, j) = el;
        }
    }
    return result;
}

Matrix Matrix::mult_par(Matrix &other, const int P) {
    Matrix result(n_rows_, other.n_cols_);

    omp_set_num_threads(P);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_rows_; ++i) {
        for (int j = 0; j < other.n_cols_; ++j) {
            double el = 0;
            for (int k = 0; k < n_cols_; ++k) {
                el += (*this)(i, k) * other(k, j);
            }
            result(i, j) = el;
        }
    }
    return result;
}

void Matrix::fill_rand() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < n_rows_; ++i) {
        for (int j = 0; j < n_cols_; ++j) {
            (*this)(i, j) = dis(gen);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Not enough arguments!" << std::endl;
        return -1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    int P = std::atoi(argv[4]);

    Matrix A(M, N);
    Matrix B(N, K);
    A.fill_rand();
    B.fill_rand();

    double start_time, end_time;
    Matrix C;

    start_time = omp_get_wtime();
    C = A * B;
    end_time = omp_get_wtime();
    double T1 = end_time - start_time;
    std::cout << "Время последовательного перемножения: " << T1 << std::endl;


    start_time = omp_get_wtime();
    C = A.mult_par(B, P);
    end_time = omp_get_wtime();
    double Tp = end_time - start_time;
    std::cout << "Время паралельного перемножения: " << Tp << std::endl;

    double S = T1 / Tp;
    std::cout << "S(speedup) = " << S << std::endl;

    double E = S / P;
    std::cout << "E = " << E << std::endl;

    double Co = P * Tp;
    double T0 = Co * T1;
    std::cout << "C(cost) = " << Co << std::endl;
    std::cout << "T0 = " << T0 << std::endl;






    return 0;
}