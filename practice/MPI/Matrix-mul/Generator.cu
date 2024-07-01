#include <fstream>
#include <iostream>
#include <random>
#include <string>

// #define DEBUG

#define ERROR 1e-6
#define DATA_PATH "./attachment/"

__global__ void matrixMul(const double *A, const double *B, double *C, int n1, int n2, int n3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    if (row < n1 && col < n3) {
        for (int i = 0; i < n2; i++) {
            sum += A[row * n2 + i] * B[i * n3 + col];
        }
        C[row * n3 + col] = sum;
    }
}

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " filename <n1> <n2> <n3>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    std::string in_filename = std::string(DATA_PATH)+filename+".data";
    std::string out_filename = std::string(DATA_PATH)+filename+".ref";

    uint64_t n1 = atoll(argv[1]);
    uint64_t n2 = atoll(argv[2]);
    uint64_t n3 = atoll(argv[3]);

    std::ofstream in(in_filename, std::ios::binary);
    std::ofstream out(out_filename, std::ios::binary);

    double *a = new double[n1 * n2];
    double *b = new double[n2 * n3];
    double *c = new double[n1 * n3];

#ifdef DEBUG
    double *test = new double[n1 * n3];
#endif

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n1 * n2 * sizeof(double));
    cudaMalloc(&d_b, n2 * n3 * sizeof(double));
    cudaMalloc(&d_c, n1 * n3 * sizeof(double));

    // Generate random data a, b
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 10);

    for (uint64_t i = 0; i < n1 * n2; i++)
    {
        a[i] = dis(gen);
    }

    for (uint64_t i = 0; i < n2 * n3; i++)
    {
        b[i] = dis(gen);
    }

#ifndef DEBUG

    in.write(reinterpret_cast<char *>(&n1), sizeof(uint64_t));
    in.write(reinterpret_cast<char *>(&n2), sizeof(uint64_t));
    in.write(reinterpret_cast<char *>(&n3), sizeof(uint64_t));

    in.write(reinterpret_cast<char *>(a), n1 * n2 * sizeof(double));
    in.write(reinterpret_cast<char *>(b), n2 * n3 * sizeof(double));

#endif

    in.close();

    cudaMemcpy(d_a, a, n1 * n2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n2 * n3 * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n3 + 15) / 16, (n1 + 15) / 16);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n1, n2, n3);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n1 * n3 * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef DEBUG

    // #pragma omp parallel for
    for (uint64_t i = 0; i < n1; i++)
    {
        for (uint64_t j = 0; j < n3; j++)
        {
            double sum = 0.0;
            for (uint64_t k = 0; k < n2; k++)
            {
                sum += a[i * n2 + k] * b[k * n3 + j];
            }
            test[i * n3 + j] = sum;
        }
    }

    for (uint64_t i = 0; i < n1 * n3; i++)
    {
        if (std::abs(c[i] - test[i]) > ERROR)
        {
            std::cerr << "Error: " << c[i] << " " << test[i] << std::endl;
            return 1;
        }
    }

    std::cout << "Correct!" << std::endl;

#else

    out.write(reinterpret_cast<char *>(&n1), sizeof(uint64_t));
    out.write(reinterpret_cast<char *>(&n3), sizeof(uint64_t));
    out.write(reinterpret_cast<char *>(c), n1 * n3 * sizeof(double));

#endif

    out.close(); 
    delete[] a;
    delete[] b;
    delete[] c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}