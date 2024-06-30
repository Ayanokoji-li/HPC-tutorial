#include <ctime>
#include <iostream>
#include <omp.h>
const int N = 1048576;
const int MOD = 256;

int main() {
  int a[N];
  int sum_ref = 0, sum = 0;
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    a[i] = std::time(0) % MOD;
  }

  double start = omp_get_wtime();
  for (int i = 0; i < N; ++i) {
    sum_ref += a[i];
  }
  double end = omp_get_wtime();
  std::cout << "Reference time: " << end - start << std::endl;

  start = omp_get_wtime();
  // Your code here.
  // Hint: try 'atomic', 'critical', 'reduction' pragmas.
  end = omp_get_wtime();
  std::cout << "OpenMP time: " << end - start << std::endl;

  if (sum_ref != sum) {
    std::cerr << "Error: " << sum_ref << " != " << sum << std::endl;
    return 1;
  }
  return 0;
}