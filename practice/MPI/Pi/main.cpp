#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>

int main()
{
    std::ofstream file("ans.bin", std::ios::binary);
    
    long double pi = 0.0;


    // Your code here
    // ###############################################




    // ###############################################

    file.write(reinterpret_cast<char*>(&pi), sizeof(pi));
    file.close();

    // You may fix the multi-output
    std::cout << std::fixed << std::setprecision(20) << "Pi = " << pi << std::endl;

    return 0;
}