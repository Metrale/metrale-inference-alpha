// SPDX-License-Identifier: AGPL-3.0-only

// nvcc -arch=sm_90 -O2 scripts/hopper-argmax-unique-check.cu -o /tmp/argmax-check
// Run only in an exclusive GPU window; this does not load a model.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include "../kernels/gb10/common/argmax_bf16.cu"
#include "../kernels/hopper/common/argmax_unique.cu"

#define CUDA_OK(expr) do { const cudaError_t e = (expr); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); std::exit(2); \
} } while (0)

int main() {
    constexpr unsigned int rows = 9, vocab = 2051, sentinel = 0xffffffffu;
    std::vector<__nv_bfloat16> input(rows * vocab, __float2bfloat16(-2.0f));
    auto put = [&](unsigned int row, unsigned int col, float value) {
        input[row * vocab + col] = __float2bfloat16(value);
    };
    put(0, 37, 2.0f);                          // Unique finite maximum.
    put(1, 0, 2.0f); put(1, 1024, 2.0f);       // Same-thread exact BF16 tie.
    put(2, 37, 2.0f); put(2, 2050, std::numeric_limits<float>::quiet_NaN());
    put(3, 37, std::numeric_limits<float>::infinity());
    put(4, 37, 2.0f); put(4, 2049, -std::numeric_limits<float>::infinity());
    put(5, 37, -0.0f); put(5, 1025, 0.0f);     // Signed zero is a tie.
    put(6, 2050, 2.0f);                        // Tail beyond two full tiles.
    put(7, 2047, 2.0f);                        // EOS winner: caller must mask.
    for (unsigned int col = 0; col < vocab; ++col) put(8, col, -3e38f);
    put(8, 2049, -2e38f);                      // All values below legacy -1e30.
    const unsigned int expected[rows] = {37, sentinel, sentinel, sentinel,
        sentinel, sentinel, 2050, 2047, 2049};
    __nv_bfloat16* device_input = nullptr;
    unsigned int* device_output = nullptr;
    CUDA_OK(cudaMalloc(&device_input, input.size() * sizeof(input[0])));
    CUDA_OK(cudaMalloc(&device_output, rows * sizeof(unsigned int)));
    CUDA_OK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice));
    std::vector<unsigned int> output(rows);
    for (int repetition = 0; repetition < 2; ++repetition) {
        CUDA_OK(cudaMemset(device_output, 0xcd, rows * sizeof(unsigned int)));
        argmax_bf16_batch_unique<<<rows, 1024>>>(device_input, device_output, vocab, vocab);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemcpy(output.data(), device_output, rows * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        for (unsigned int row = 0; row < rows; ++row) {
            if (output[row] != expected[row]) {
                std::fprintf(stderr, "FAIL row=%u repetition=%d got=%u expected=%u\n", row, repetition, output[row], expected[row]);
                return 1;
            }
        }
    }
    // Known-bad control: old argmax cannot certify host LAST-index tie semantics.
    argmax_bf16_batch<<<1, 1024>>>(device_input + vocab, device_output, vocab, vocab);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(output.data(), device_output, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (output[0] != 0 || output[0] == 1024) {
        std::fprintf(stderr, "FAIL legacy tie oracle got=%u\n", output[0]);
        return 1;
    }
    CUDA_OK(cudaFree(device_output));
    CUDA_OK(cudaFree(device_input));
    std::puts("PASS unique/tie/NaN/+inf/-inf/signed-zero/tail/EOS/negative/multirow/repeated-overwrite; legacy tie divergence reproduced");
    return 0;
}
