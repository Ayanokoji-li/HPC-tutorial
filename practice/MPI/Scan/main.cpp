#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>
#include <chrono>
#include <mpi.h>

#define err 1e-5

void Naive_Scan(double *data, double *ans, long N)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *buffer = new double[N];
    if(rank > 0)
    {
        MPI_Recv(buffer, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (long i = 0; i < N; i++)
        {
            ans[i] = buffer[i] + data[i];
        }
    }
    else
    {
        std::memcpy(ans, data, N * sizeof(double));
    }

    if(rank < size - 1)
    {
        MPI_Send(ans, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }
}

void Yours_Scan(double *data, double *ans, long N)
{
    return;
}

int main(int argc, char const *argv[])
{
    long N_per_rank = 100;
    int loop_time = 10;

    if(argc == 3)
    {
        loop_time = std::stoi(argv[1]);
        N_per_rank = std::stol(argv[2]);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " <loop_time> <N_per_rank>" << std::endl;
        return 1;
    }

    double *MPI_data = new double[N_per_rank];
    double *MPI_ans = new double[N_per_rank];
    double *Naive_data = new double[N_per_rank];
    double *Naive_ans = new double[N_per_rank];
    double *Your_data = new double[N_per_rank];
    double *Your_ans = new double[N_per_rank];
    int rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (long i = 0; i < N_per_rank; i++)
    {
        MPI_data[i] = dis(gen);
    }
    std::memcpy(Naive_data, MPI_data, N_per_rank * sizeof(double));
    std::memcpy(Your_data, MPI_data, N_per_rank * sizeof(double));

    // check correctness
    MPI_Scan(MPI_data, MPI_ans, N_per_rank, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Naive_Scan(Naive_data, Naive_ans, N_per_rank);
    Yours_Scan(Your_data, Your_ans, N_per_rank);

    bool correct = true;
    for (long i = 0; i < N_per_rank; i++)
    {
        if(std::abs(MPI_ans[i] - Your_ans[i]) > err)
        {
            correct = false;
            break;
        }
    }

    bool total_correct = correct;    
    MPI_Allreduce(&correct, &total_correct, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if(rank == 0)
    {    
        if(correct)
        {
            std::cout << "Correct!" << std::endl;
        }
        else
        {
            std::cout << "Incorrect!" << std::endl;
        }
    }
    
    // benchmark
    if(total_correct)
    {        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < loop_time; i++)
        {
            MPI_Scan(MPI_data, MPI_ans, N_per_rank, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto MPI_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)loop_time ;

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < loop_time; i++)
        {
            Naive_Scan(Naive_data, Naive_ans, N_per_rank);
        }
        end = std::chrono::high_resolution_clock::now();

        auto Naive_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)loop_time ;

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < loop_time; i++)
        {
            Yours_Scan(Your_data, Your_ans, N_per_rank);
        }
        end = std::chrono::high_resolution_clock::now();

        auto Your_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)loop_time ;

        if(rank == 0)
        {
            std::cout << "MPI: " << MPI_duration << " ms" << std::endl;
            std::cout << "Naive: " << Naive_duration << " ms" << std::endl;
            std::cout << "Your: " << Your_duration << " ms" << std::endl;
        }
    }

    MPI_Finalize();

    delete[] MPI_data;
    delete[] MPI_ans;
    delete[] Naive_data;
    delete[] Naive_ans;
    delete[] Your_data;
    delete[] Your_ans;

    return 0;
}
