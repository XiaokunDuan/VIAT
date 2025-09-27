cd /hy-tmp
cat <<EOF > test.cu
#include <iostream>

__global__ void kernel() {
}

int main() {
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return -1;
    }
    std::cout << "Minimal CUDA C++ program compiled and ran successfully!" << std::endl;
    return 0;
}
EOF