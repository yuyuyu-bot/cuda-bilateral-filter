#ifndef DEVICE_UTILITIES_CUH
#define DEVICE_UTILITIES_CUH

constexpr auto kernel_color_table_len = 256 * 3;

template <typename ValueType>
inline __device__ ValueType clamp(const ValueType v, const ValueType min, const ValueType max) {
    return v < min ? min :
           v > max ? max :
           v;
}

#endif // DEVICE_UTILITIES_CUH
