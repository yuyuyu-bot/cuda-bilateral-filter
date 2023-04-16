# cuda-bilateral-filter

## やりたいこと

CUDAでバイラテラルフィルタを高速化する（リハビリ）。\
カリカリにチューニングするわけではない、インターンのCUDAチュートリアルレベル。

なお、動作に用いる画像は512x512のRGB画像 ([lenna](sample_image/lenna.png))。\
また、100回実行の平均を実行時間として記載、カーネルサイズは15固定。

## 1: ナイーブな実装

src: [1-naive.cuh](include/1-naive.cuh)

```cpp
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
```

実行時間とOpenCV (C++)実装との誤差は以下の通り。
```
OpenCV:
        duration  : 17686 [usec]
Naive:
        diff count: 1
        max diff  : 1
        duration  : 1041.28 [usec]
```
当たり前にOpenCVよりは速い。

## 2: カーネルを事前計算

カーネル係数は（入力画像が整数値しかとらない場合）事前計算が可能なため、スレッド内処理の外で先に計算しておく。
```cpp

const auto get_kernel_space = [ksize, radius, kernel_space](const int kx, const int ky) {
    return kernel_space[(ky + radius) * ksize + (kx + radius)];  // 事前に計算しておく
};

const auto get_kernel_color = [kernel_color_table](const auto a, const auto b) {
    const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
    const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
    const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
    const auto color_distance = abs(diff0) + abs(diff1) + abs(diff2);
    return kernel_color_table[color_distance];  // 事前に計算しておく
};
```

遅くなった。
```
OpenCV:
        duration  : 17686 [usec]
Naive:
        diff count: 1
        max diff  : 1
        duration  : 1041.28 [usec]
Precompute kernel:
        diff count: 1
        max diff  : 1
        duration  : 6143.51 [usec]
```
事前計算しておいたテーブルをグローバルメモリに置いているので遅くなるだろうな、とは思っていたがこんなに遅くなるのか。

## 3: カーネルをshared memory上に置く

事前計算したものをグローバルメモリに置いていると当然遅いので、shared memoryの上に置くことにする。
```cpp

extern __shared__ float s_bilateral_filter_buffer[];
auto s_kernel_space       = &s_bilateral_filter_buffer[0];
auto s_kernel_color_table = &s_bilateral_filter_buffer[ksize * ksize];

// shared memoryに事前計算の結果をコピー
for (int i = blockDim.x * threadIdx.y + threadIdx.x; i < ksize * ksize; i += blockDim.x * blockDim.y) {
    s_kernel_space[i] = kernel_space[i];
}
for (int i = blockDim.x * threadIdx.y + threadIdx.x; i < kernel_color_table_len; i += blockDim.x * blockDim.y) {
    s_kernel_color_table[i] = kernel_color_table[i];
}
__syncthreads();

const auto get_kernel_space = [ksize, radius, s_kernel_space](const int kx, const int ky) {
    return s_kernel_space[(ky + radius) * ksize + (kx + radius)];  // shared memory上の方を読みに行く
};

const auto get_kernel_color = [s_kernel_color_table](const auto a, const auto b) {
    const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
    const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
    const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
    const auto color_distance = abs(diff0) + abs(diff1) + abs(diff2);
    return s_kernel_color_table[color_distance];  // shared memory上の方を読みに行く
};
```

速くなったが、ナイーブ実装に比べるとそこまで速くはなっていない。
```
OpenCV:
        duration  : 17686 [usec]
Naive:
        diff count: 1
        max diff  : 1
        duration  : 1041.28 [usec]
Precompute kernel:
        diff count: 1
        max diff  : 1
        duration  : 6143.51 [usec]
Kernel on shared memory:
        diff count: 1
        max diff  : 1
        duration  : 945.251 [usec]
```

ナイーブ実装との差分は「`exp`を呼ぶかshared memoryを読みに行くか」だけなのでそこまで差が出るものではないか。

## 4: 入力画像もshared memoryに置く

2,3でグローバルメモリアクセスがあまりに遅いことは分かった。\
となると入力画像を読みに行くところをどうにかしたい気持ちになる。
```cpp
extern __shared__ float s_bilateral_filter_buffer[];
auto s_kernel_space       = &s_bilateral_filter_buffer[0];
auto s_kernel_color_table = &s_bilateral_filter_buffer[ksize * ksize];

const int smem_width    = blockDim.x + ksize - 1;
const int smem_height   = blockDim.y + ksize - 1;
const int smem_stride   = smem_width * 3;
const int smem_origin_x = x - tx - radius;
const int smem_origin_y = y - ty - radius;
auto s_src              = reinterpret_cast<std::uint8_t*>(&s_kernel_color_table[kernel_color_table_len]);

for (int i = blockDim.x * threadIdx.y + threadIdx.x; i < ksize * ksize; i += blockDim.x * blockDim.y) {
    s_kernel_space[i] = kernel_space[i];
}
for (int i = blockDim.x * threadIdx.y + threadIdx.x; i < kernel_color_table_len; i += blockDim.x * blockDim.y) {
    s_kernel_color_table[i] = kernel_color_table[i];
}

const auto get_s_src_ptr = [s_src, smem_stride, smem_origin_x, smem_origin_y](const int src_x, const int src_y) {
    const auto s_src_x = src_x - smem_origin_x;
    const auto s_src_y = src_y - smem_origin_y;
    return &s_src[smem_stride * s_src_y + s_src_x * 3];
};

// 入力画像をshared memoryにコピー
for (int y_offset = ty; y_offset < smem_height; y_offset += blockDim.y) {
    for (int x_offset = tx; x_offset < smem_width; x_offset += blockDim.x) {
        auto* const s_src_ptr = get_s_src_ptr(smem_origin_x + x_offset, smem_origin_y + y_offset);
        const auto x_clamped = clamp(smem_origin_x + x_offset, 0, width - 1);
        const auto y_clamped = clamp(smem_origin_y + y_offset, 0, height - 1);
        s_src_ptr[0] = src[stride_3ch * y_clamped + x_clamped * 3 + 0];
        s_src_ptr[1] = src[stride_3ch * y_clamped + x_clamped * 3 + 1];
        s_src_ptr[2] = src[stride_3ch * y_clamped + x_clamped * 3 + 2];
    }
}
__syncthreads();
```

一気に速くなった。
```
OpenCV:
        duration  : 17686 [usec]
Naive:
        diff count: 1
        max diff  : 1
        duration  : 1041.28 [usec]
Precompute kernel:
        diff count: 1
        max diff  : 1
        duration  : 6143.51 [usec]
Kernel on shared memory:
        diff count: 1
        max diff  : 1
        duration  : 945.251 [usec]
Image on shared memory:
        diff count: 1
        max diff  : 1
        duration  : 318.42 [usec]
```
OpenCVの55倍、ナイーブ実装の3倍。

## まとめ・所感

とりあえずパッと思いつたところを順を追って高速化してみた。\
グローバルメモリにばかばかアクセスするのは悪だということを再認識した。
