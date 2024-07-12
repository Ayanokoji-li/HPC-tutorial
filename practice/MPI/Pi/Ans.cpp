#include <fstream>
#include <iostream>
#include <iomanip>

#define N 1000000000000

int main() {
    long double pi = 0.0;
    long double h = 1.0L / N;
    #pragma omp parallel for reduction(+:pi)
    for (long long i = 0; i < N; i++) {
        long double x = (i + 0.5) * h;
        pi += 4.0 / (1.0 + x * x);
    }
    pi *= h;
    std::cout << std::fixed << std::setprecision(20) << "Pi = " << pi << std::endl;

    // Write to file by binary
    std::ofstream file("pi.bin", std::ios::binary);
    file.write(reinterpret_cast<char*>(&pi), sizeof(pi));
    file.close();
    
    return 0;
}