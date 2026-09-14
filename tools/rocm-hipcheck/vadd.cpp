// Minimal HIP kernel: confirms the spoofed gfx1030 target actually EXECUTES on the 680M.
// rocminfo enumerating an agent is necessary but not sufficient - kernels must run.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int dev = 0;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        printf("FAIL: hipGetDeviceProperties\n");
        return 1;
    }
    printf("device            : %s\n", prop.name);
    printf("gcnArchName       : %s\n", prop.gcnArchName);
    printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf("memoryClockRate   : %.0f MHz\n", prop.memoryClockRate / 1000.0);
    printf("memoryBusWidth    : %d bits\n", prop.memoryBusWidth);
    printf("totalGlobalMem    : %.2f GB\n", prop.totalGlobalMem / 1073741824.0);

    const int n = 1 << 20;
    std::vector<float> ha(n, 1.5f), hb(n, 2.25f), hc(n, 0.0f);
    float *da, *db, *dc;
    hipMalloc(&da, n * sizeof(float));
    hipMalloc(&db, n * sizeof(float));
    hipMalloc(&dc, n * sizeof(float));
    hipMemcpy(da, ha.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(db, hb.data(), n * sizeof(float), hipMemcpyHostToDevice);

    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0);
    for (int rep = 0; rep < 50; ++rep)
        hipLaunchKernelGGL(vadd, dim3(n / 256), dim3(256), 0, 0, da, db, dc, n);
    hipDeviceSynchronize();
    hipEventRecord(t1);
    hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);

    hipMemcpy(hc.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost);
    double err = 0;
    for (int i = 0; i < n; ++i) err += (hc[i] != 3.75f);
    printf("vadd result      : %s (mismatches %ld)\n", err == 0 ? "CORRECT" : "WRONG", (long)err);
    printf("vadd 50x %d add  : %.3f ms total\n", n, ms);
    double gbps = 50.0 * 3.0 * n * sizeof(float) / (ms / 1000.0) / 1e9;
    printf("effective bandwidth: %.1f GB/s\n", gbps);
    return 0;
}
