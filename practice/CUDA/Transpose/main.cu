#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>

__global__ void warmupKernel() {
    return;
}

__global__ void transposeOptimized(float *odata, float *idata, int width, int height){
    return;
}
    

__global__ void transposeNaive(float *odata, float *idata, int width,int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        odata[x * height + y] = idata[y * width + x];
    }
}

void transpose(float *odata, float *idata, int width, int height) {
    #pragma omp parallel for
    for(int x = 0; x < width; x++) {
        for(int y = 0; y < height; y++) {
            odata[x * height + y] = idata[y * width + x];
        }
    }
}

int main(int argc, char const *argv[])
{
    int width = 1 << 10;
    int height = 1 << 10;
    int loop_times = 10;

    if(argc == 4)
    {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
        loop_times = atoi(argv[3]);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " <width> <height> <loop_times>" << std::endl;
        return 1;
    }

    float *h_idata = new float[width * height];
    float *h_odata = new float[width * height];
    float *h_odata2 = new float[width * height];

    warmupKernel<<<1, 1>>>();

    float *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, width * height * sizeof(float));
    cudaMalloc((void **)&d_odata, width * height * sizeof(float));

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for(int i = 0; i < width * height; i++) {
        h_idata[i] = distribution(generator);
    }
    transpose(h_odata, h_idata, width, height);
    cudaMemcpy(d_idata, h_idata, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // modify the block size and grid size
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // check correctness
    transposeOptimized<<<gridSize, blockSize>>>(d_odata, d_idata, width, height);
    cudaMemcpy(h_odata2, d_odata, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for(int i = 0; i < width * height; i++) {
        if(h_odata[i] != h_odata2[i]) {
            correct = false;
            break;
        }
    }

    if(correct)
    {
        std::cout << "Correct!" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        // for(int i = 0; i < loop_times; i++) {
        //     transpose(h_odata, h_idata, width, height);
        // }
        auto end = std::chrono::high_resolution_clock::now();
        auto CPU_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / loop_times;

        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < loop_times; i++) {
            transposeNaive<<<gridSize, blockSize>>>(d_odata, d_idata, width, height);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto GPU_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / loop_times;

        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < loop_times; i++) {
            transposeOptimized<<<gridSize, blockSize>>>(d_odata, d_idata, width, height);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto GPU_time_optimized = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / loop_times;

        // std::cout << "CPU time: " << CPU_time << " ms" << std::endl;
        std::cout << "GPU time: " << GPU_time << " ms" << std::endl;
        std::cout << "Yours time: " << GPU_time_optimized << " ms" << std::endl;
    }
    else
    {
        std::cout << "Incorrect!" << std::endl;
    }

    delete[] h_idata;
    delete[] h_odata;
    delete[] h_odata2;
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}
