#ifndef DEVICE_UTILITIES_CUH
#define DEVICE_UTILITIES_CUH

#include <thrust/device_vector.h>

constexpr auto kernel_color_table_len = 256 * 3;

template <typename ValueType>
inline __device__ ValueType clamp(const ValueType v, const ValueType min, const ValueType max) {
    return v < min ? min :
           v > max ? max :
           v;
}

inline auto precompute_kernels(
    const int ksize,
    const float sigma_space,
    const float sigma_color,
    thrust::device_vector<float>& d_kernel_space,
    thrust::device_vector<float>& d_kernel_color_table
) {
    const int radius  = ksize / 2;
    const auto gauss_space_coeff = -1.f / (2 * sigma_space * sigma_space);
    const auto gauss_color_coeff = -1.f / (2 * sigma_color * sigma_color);

    std::vector<float> h_kernel_space(ksize * ksize);
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto kidx = (ky + radius) * ksize + (kx + radius);
            const auto r2 = kx * kx + ky * ky;
            if (r2 > radius * radius) {
                continue;
            }
            h_kernel_space[kidx] = std::exp(r2 * gauss_space_coeff);
        }
    }
    thrust::copy(h_kernel_space.begin(), h_kernel_space.end(), d_kernel_space.begin());

    std::vector<float> h_kernel_color_table(kernel_color_table_len);
    for (std::uint32_t i = 0; i < h_kernel_color_table.size(); i++) {
        h_kernel_color_table[i] = std::exp((i * i) * gauss_color_coeff);
    }
    thrust::copy(h_kernel_color_table.begin(), h_kernel_color_table.end(), d_kernel_color_table.begin());
}

#endif // DEVICE_UTILITIES_CUH
