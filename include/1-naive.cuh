#ifndef NAIVE_CUH
#define NAIVE_CUH

#include <cstdint>
#include "common.cuh"

namespace bilateral_filter {

__global__ void naive_kernel(
    const std::uint8_t* const src,
    std::uint8_t* const       dst,
    const int                 width,
    const int                 height,
    const int                 ksize,
    const float               sigma_space,
    const float               sigma_color
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int stride_3ch = width * 3;
    const int radius  = ksize / 2;
    const auto gauss_space_coeff = -1.f / (2 * sigma_space * sigma_space);
    const auto gauss_color_coeff = -1.f / (2 * sigma_color * sigma_color);

    if (x >= width || y >= height) {
        return;
    }

    const auto get_kernel_space = [radius, gauss_space_coeff](const auto r2) {
        return expf(r2 * gauss_space_coeff);
    };

    const auto get_kernel_color = [gauss_color_coeff](const auto a, const auto b) {
        const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
        const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
        const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
        const auto color_distance = abs(diff0) + abs(diff1) + abs(diff2);
        return expf((color_distance * color_distance) * gauss_color_coeff);
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

            const auto kernel_space = get_kernel_space(r2);
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

void naive(
    const std::uint8_t* const d_src,
    std::uint8_t* const       d_dst,
    const int                 width,
    const int                 height,
    const int                 ksize,
    const float               sigma_space,
    const float               sigma_color
) {
    constexpr std::uint32_t block_width  = 32u;
    constexpr std::uint32_t block_height = 32u;
    const std::uint32_t grid_width   = (width  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height + block_height - 1) / block_height;

    const dim3 grid_dim(grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    naive_kernel<<<grid_dim, block_dim>>>(d_src, d_dst, width, height, ksize, sigma_space, sigma_color);
    cudaDeviceSynchronize();
}

}  // namespace bilateral_filter

#endif  // NAIVE_CUH
