#ifndef PRECOMPUTE_KERNEL
#define PRECOMPUTE_KERNEL

#include <cstdint>
#include <thrust/device_vector.h>
#include <vector>
#include "common.cuh"

namespace bilateral_filter {

__global__ void precompute_kernel_kernel(
    const std::uint8_t* const src,
    std::uint8_t* const       dst,
    const int                 width,
    const int                 height,
    const int                 ksize,
    const float* const        kernel_space,
    const float* const        kernel_color_table
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int stride_3ch = width * 3;
    const int radius  = ksize / 2;

    if (x >= width || y >= height) {
        return;
    }

    const auto get_kernel_space = [ksize, radius, kernel_space](const int kx, const int ky) {
        return kernel_space[(ky + radius) * ksize + (kx + radius)];
    };

    const auto get_kernel_color = [kernel_color_table](const auto a, const auto b) {
        const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
        const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
        const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
        const auto color_distance = abs(diff0) + abs(diff1) + abs(diff2);
        return kernel_color_table[color_distance];
    };

    const auto src_center_pix = src + stride_3ch * y + x * 3;
    auto sum0 = 0.f;
    auto sum1 = 0.f;
    auto sum2 = 0.f;
    auto sumk = 0.f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto r2 = kx * kx + ky * ky;
            if (r2 > radius * radius) {
                continue;
            }

            const auto x_clamped = clamp(x + kx, 0, width - 1);
            const auto y_clamped = clamp(y + ky, 0, height - 1);
            const auto src_pix = src + stride_3ch * y_clamped + x_clamped * 3;

            const auto kernel_space = get_kernel_space(kx, ky);
            const auto kernel_color = get_kernel_color(src_center_pix, src_pix);
            const auto kernel = kernel_space * kernel_color;

            sum0 += src_pix[0] * kernel;
            sum1 += src_pix[1] * kernel;
            sum2 += src_pix[2] * kernel;
            sumk += kernel;
        }
    }

    dst[stride_3ch * y + x * 3 + 0] = static_cast<std::uint8_t>(sum0 / sumk + 0.5f);
    dst[stride_3ch * y + x * 3 + 1] = static_cast<std::uint8_t>(sum1 / sumk + 0.5f);
    dst[stride_3ch * y + x * 3 + 2] = static_cast<std::uint8_t>(sum2 / sumk + 0.5f);
}

void precompute_kernel(
    const std::uint8_t* const d_src,
    std::uint8_t* const       d_dst,
    const int                 width,
    const int                 height,
    const int                 ksize,
    const float               sigma_space,
    const float               sigma_color
) {
    static bool initialized = false;
    static thrust::device_vector<float> d_kernel_space(ksize * ksize);
    static thrust::device_vector<float> d_kernel_color_table(kernel_color_table_len);

    if (!initialized) {
        precompute_kernels(ksize, sigma_space, sigma_color, d_kernel_space, d_kernel_color_table);
        initialized = true;
    }

    constexpr std::uint32_t block_width  = 32u;
    constexpr std::uint32_t block_height = 32u;
    const std::uint32_t grid_width   = (width  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height + block_height - 1) / block_height;

    const dim3 grid_dim(grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    precompute_kernel_kernel<<<grid_dim, block_dim>>>(
        d_src, d_dst, width, height, ksize, d_kernel_space.data().get(), d_kernel_color_table.data().get());
    cudaDeviceSynchronize();
}

}  // namespace bilateral_filter

#endif  // PRECOMPUTE_KERNEL
