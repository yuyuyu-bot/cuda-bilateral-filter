#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thrust/device_vector.h>

#include "1-naive.cuh"
#include "2-precompute_kernel.cuh"
#include "3-kernel_on_shared_memory.cuh"
#include "4-image_on_shared_memory.cuh"

#define BENCHMARK(fn, duration)                                                                                        \
do {                                                                                                                   \
    constexpr int itr = 100;                                                                                           \
    std::int64_t sum_duration = 0ll;                                                                                   \
    for (int i = 0; i <= itr; i++) {                                                                                   \
        const auto start = std::chrono::system_clock::now();                                                           \
        (fn);                                                                                                          \
        const auto end = std::chrono::system_clock::now();                                                             \
                                                                                                                       \
        if (i != 0) {                                                                                                  \
            sum_duration  += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();                \
        }                                                                                                              \
    }                                                                                                                  \
    duration = sum_duration / itr / 1e3f;                                                                              \
} while (false)

template <typename ElemType>
class DeviceBuffer {
public:
    DeviceBuffer(const std::size_t len) : data_(len) {}

    void upload(const ElemType* const data) {
        thrust::copy(data, data + data_.size(), data_.begin());
    }

    void download(ElemType* const data) {
        thrust::copy(data_.begin(), data_.end(), data);
    }

    ElemType* get() {
        return data_.data().get();
    }

    const ElemType* get() const {
        return data_.data().get();
    }

private:
    thrust::device_vector<ElemType> data_;
};

auto count_diff(const cv::Mat3b& a, const cv::Mat3b& b) {
    assert(a.size() == b.size());
    auto count = 0;
    auto max = 0;
    for (int y = 0; y < a.rows; y++) {
        for (int x = 0; x < a.cols; x++) {
            const auto diff0 = static_cast<int>(a(y, x)[0]) - static_cast<int>(b(y, x)[0]);
            const auto diff1 = static_cast<int>(a(y, x)[1]) - static_cast<int>(b(y, x)[1]);
            const auto diff2 = static_cast<int>(a(y, x)[2]) - static_cast<int>(b(y, x)[2]);
            if (diff0 > 0 || diff1 > 0 || diff2 > 0) {
                count++;
                max = std::max({max, diff0, diff1, diff2});
            }
        }
    }

    return std::make_tuple(count, max);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "[Usage] " << argv[0] << " filename [ksize] [sigma_space] [sigma_color]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const auto filename    = std::string(argv[1]);
    const auto ksize       = argc >= 3 ? std::stoi(argv[2]) : 15;
    const auto sigma_space = argc >= 4 ? std::stof(argv[3]) : 30.f;
    const auto sigma_color = argc >= 5 ? std::stof(argv[4]) : 30.f;

    const cv::Mat input_image = cv::imread(filename, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Failed to load " << filename << std::endl;
        return 0;
    }
    const int width = input_image.cols;
    const int height = input_image.rows;
    const int image_length = width * height * 3;
    DeviceBuffer<std::uint8_t> d_input_image(image_length);
    d_input_image.upload(input_image.ptr<std::uint8_t>());


    // opencv
    cv::Mat3b dst_opencv;
    {
        float duration;
        BENCHMARK(cv::bilateralFilter(input_image, dst_opencv, ksize, sigma_color, sigma_space, cv::BORDER_REPLICATE), duration);

        std::cout << "OpenCV:" << std::endl;
        std::cout << "\tduration  : " << duration << " [usec]" << std::endl;
    }

    // cuda naive
    {
        DeviceBuffer<std::uint8_t> d_dst(image_length);
        cv::Mat3b dst(input_image.size());

        // execute
        float duration;
        BENCHMARK(bilateral_filter::naive(d_input_image.get(), d_dst.get(), width, height, ksize, sigma_color, sigma_space), duration);

        d_dst.download(dst.ptr<std::uint8_t>());

        const auto [count, max_diff] = count_diff(dst_opencv, dst);
        std::cout << "Naive:" << std::endl;
        std::cout << "\tdiff count: " << count << std::endl;
        std::cout << "\tmax diff  : " << max_diff << std::endl;
        std::cout << "\tduration  : " << duration << " [usec]" << std::endl;
    }

    // cuda precompute kernel
    {
        DeviceBuffer<std::uint8_t> d_dst(image_length);
        cv::Mat3b dst(input_image.size());

        // execute
        float duration;
        BENCHMARK(bilateral_filter::precompute_kernel(d_input_image.get(), d_dst.get(), width, height, ksize, sigma_color, sigma_space), duration);

        d_dst.download(dst.ptr<std::uint8_t>());

        const auto [count, max_diff] = count_diff(dst_opencv, dst);
        std::cout << "Precompute kernel:" << std::endl;
        std::cout << "\tdiff count: " << count << std::endl;
        std::cout << "\tmax diff  : " << max_diff << std::endl;
        std::cout << "\tduration  : " << duration << " [usec]" << std::endl;
    }

    // cuda kernel on shared memory
    {
        DeviceBuffer<std::uint8_t> d_dst(image_length);
        cv::Mat3b dst(input_image.size());

        // execute
        float duration;
        BENCHMARK(bilateral_filter::kernel_on_shared_memory(d_input_image.get(), d_dst.get(), width, height, ksize, sigma_color, sigma_space), duration);

        d_dst.download(dst.ptr<std::uint8_t>());

        const auto [count, max_diff] = count_diff(dst_opencv, dst);
        std::cout << "Kernel on shared memory:" << std::endl;
        std::cout << "\tdiff count: " << count << std::endl;
        std::cout << "\tmax diff  : " << max_diff << std::endl;
        std::cout << "\tduration  : " << duration << " [usec]" << std::endl;
    }

    // cuda image on shared memory
    {
        DeviceBuffer<std::uint8_t> d_dst(image_length);
        cv::Mat3b dst(input_image.size());

        // execute
        float duration;
        BENCHMARK(bilateral_filter::image_on_shared_memory(d_input_image.get(), d_dst.get(), width, height, ksize, sigma_color, sigma_space), duration);

        d_dst.download(dst.ptr<std::uint8_t>());

        const auto [count, max_diff] = count_diff(dst_opencv, dst);
        std::cout << "Image on shared memory:" << std::endl;
        std::cout << "\tdiff count: " << count << std::endl;
        std::cout << "\tmax diff  : " << max_diff << std::endl;
        std::cout << "\tduration  : " << duration << " [usec]" << std::endl;
    }

    return 0;
}
