#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>

__global__ void warmupKernel() 
{
  return;
}

void RedBlack(float *data, int size, int loop_times) 
{
  int k = 0;
  #pragma omp parallel num_threads(8)
  {
    for(; k < loop_times;)
    {
      int index = k & 1;
      
      #pragma omp for nowait
      for(int i = 1; i < size - 1; i += 2)
      {
        for(int j = 1 + index; j < size-1; j += 2)
        {
          data[i * size + j] = (data[(i - 1) * size + j] + data[(i + 1) * size + j] + data[i * size + j + 1] + data[i * size + j - 1]) / 4.0f;
        }
      }

      #pragma omp for nowait
      for(int i = 2; i < size - 1; i += 2)
      {
        for(int j = 2 - index; j < size-1; j += 2)
        {
          data[i * size + j] = (data[(i - 1) * size + j] + data[(i + 1) * size + j] + data[i * size + j + 1] + data[i * size + j - 1]) / 4.0f;
        }
      }

      #pragma omp single
      {
        k += 1;
      }
    }
  }
}

__global__ void NaiveRedBlack_onepass(float *data, int size, int index) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(i < size - 1 && j < size - 1 && i > 0 && j > 0)
  {
    if((i + j) % 2 == index)
    {
        data[i * size + j] = (data[(i - 1) * size + j] + data[(i + 1) * size + j] + data[i * size + j + 1] + data[i * size + j - 1]) / 4.0f;
    }
  }
}

void NaiveRedBlack(float *d_data, int size, int loop_times) 
{
  dim3 block(32, 32);
  dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

  for(int i = 0; i < loop_times; i++)
  {
    NaiveRedBlack_onepass<<<grid, block>>>(d_data, size, i & 1);
  }
}

void Yours(float *data, float *d_data, int size, int loop_times)
{
  return;
}

int main(int argc, char const *argv[])
{
  int size = 1;
  int loop_times = 10;

  if(argc == 3)
  {
    size = atoi(argv[1]);
    loop_times = atoi(argv[2]);
  }
  else
  {
    std::cout << "Usage: " << argv[0] << " <size> <loop_times>" << std::endl;
    return 1;
  }
  warmupKernel<<<1, 1>>>();

  float *raw_data = new float[size * size];
  float *h_data = new float[size * size];
  float *d_ans = new float[size * size];

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 10.0);
  for(int i = 0; i < size * size; i++) 
  {
    raw_data[i] = distribution(generator);
    h_data[i] = raw_data[i];
  }


  float *d_data;
  cudaMalloc((void **)&d_data, size * size * sizeof(float));
  cudaMemcpy(d_data, raw_data, size * size * sizeof(float), cudaMemcpyHostToDevice);

  NaiveRedBlack(d_data, size, loop_times);

  // check correctness
  bool correct = true;
  RedBlack(h_data, size, loop_times);
  cudaMemcpy(d_ans, d_data, size * size * sizeof(float), cudaMemcpyDeviceToHost);

  for(int i = 0; i < size * size; i++)
  {
    if(h_data[i] != d_ans[i])
    {
        correct = false;
        break;
    }
  }

  if(correct)
  {
    std::cout << "Correct!" << std::endl;

    memcpy(h_data, raw_data, size * size * sizeof(float));
    auto start = std::chrono::high_resolution_clock::now();
    RedBlack(h_data, size, loop_times);
    auto end = std::chrono::high_resolution_clock::now();
    auto CPU_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    cudaMemcpy(d_data, raw_data, size * size * sizeof(float), cudaMemcpyHostToDevice);
    start = std::chrono::high_resolution_clock::now();
    NaiveRedBlack(d_data, size, loop_times);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto Naive_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    cudaMemcpy(d_data, raw_data, size * size * sizeof(float), cudaMemcpyHostToDevice);
    start = std::chrono::high_resolution_clock::now();
    Yours(h_data, d_data, size, loop_times);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto GPU_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "CPU time: " << CPU_time << " ms" << std::endl;
    std::cout << "Naive time: " << Naive_time << " ms" << std::endl;
    std::cout << "Yours time: " << GPU_time << " ms" << std::endl;
  }
  else
  {
    std::cout << "Wrong!" << std::endl;
  }

  delete[] h_data;
  delete[] raw_data;
  delete[] d_ans;
  cudaFree(d_data);
  return 0;
}