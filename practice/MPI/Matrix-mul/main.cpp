#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <mpi.h>

#define DATA_PATH "./attachment/"

void mul(double *a, double *b, double *c, uint64_t n1, uint64_t n2, uint64_t n3)
{
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            for (int k = 0; k < n3; k++)
            {
                c[i * n3 + k] += a[i * n2 + j] * b[j * n3 + k];
            }
        }
    }
}

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }
    MPI_Init(NULL, NULL);
    double start = MPI_Wtime();

// ###########################################

    std::string filename = argv[1];
    uint64_t n1, n2, n3;
    FILE *fi;

    int rank, size;
    int dump;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    fi = fopen((std::string(DATA_PATH)+filename+".data").c_str(), "rb");
    dump = fread(&n1, 1, 8, fi);
    dump = fread(&n2, 1, 8, fi);
    dump = fread(&n3, 1, 8, fi);

    double *a = (double *)malloc(n1 * n2 * 8);
    double *b = (double *)malloc(n2 * n3 * 8);
    double *c = (double *)malloc(n1 * n3 * 8);

    dump = fread(a, 1, n1 * n2 * 8, fi);
    dump = fread(b, 1, n2 * n3 * 8, fi);
    fclose(fi);

    for (uint64_t i = 0; i < n1; i++)
    {
        for (uint64_t k = 0; k < n3; k++)
        {
            c[i * n3 + k] = 0;
        }
    }

    mul(a, b, c, n1, n2, n3);

    
    if(rank == 0)
    {
        fi = fopen((std::string(DATA_PATH)+filename+".ans").c_str(), "wb");
        dump = fwrite(c, 1, n1 * n3 * 8, fi);
        fclose(fi);
    }

    free(a);
    free(b);
    free(c);

// ###########################################

    double end = MPI_Wtime();
    if(rank == 0)
    {
        std::cout << "Time: " << end - start << std::endl;
    }
    MPI_Finalize();
    return 0;
}
