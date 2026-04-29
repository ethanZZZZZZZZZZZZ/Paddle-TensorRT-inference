#include "cuda/preprocess_kernel.h"

#ifdef EDGE_ENABLE_CUDA

#include <cuda_runtime.h>

namespace edge {
namespace {

__device__ float ClampFloat(float value, float low, float high) {
    return fminf(fmaxf(value, low), high);
}

__device__ float SampleBilinear(
    const uint8_t* image,
    int width,
    int height,
    int stride_bytes,
    float src_x,
    float src_y,
    int channel) {
    src_x = ClampFloat(src_x, 0.0F, static_cast<float>(width - 1));
    src_y = ClampFloat(src_y, 0.0F, static_cast<float>(height - 1));

    const int x0 = static_cast<int>(floorf(src_x));
    const int y0 = static_cast<int>(floorf(src_y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float wx = src_x - static_cast<float>(x0);
    const float wy = src_y - static_cast<float>(y0);

    const uint8_t* row0 = image + static_cast<size_t>(y0) * static_cast<size_t>(stride_bytes);
    const uint8_t* row1 = image + static_cast<size_t>(y1) * static_cast<size_t>(stride_bytes);

    const float v00 = static_cast<float>(row0[x0 * 3 + channel]);
    const float v01 = static_cast<float>(row0[x1 * 3 + channel]);
    const float v10 = static_cast<float>(row1[x0 * 3 + channel]);
    const float v11 = static_cast<float>(row1[x1 * 3 + channel]);

    const float top = v00 * (1.0F - wx) + v01 * wx;
    const float bottom = v10 * (1.0F - wx) + v11 * wx;
    return top * (1.0F - wy) + bottom * wy;
}

__global__ void PreprocessKernel(
    const uint8_t* input_bgr,
    int input_width,
    int input_height,
    int input_stride_bytes,
    float* output_nchw,
    int output_width,
    int output_height,
    float scale,
    int pad_x,
    int pad_y,
    int batch_index,
    int batch_size) {
    const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = output_width * output_height;
    if (pixel_index >= pixels) {
        return;
    }

    const int x = pixel_index % output_width;
    const int y = pixel_index / output_width;
    const int plane_size = output_width * output_height;
    const int resized_width = max(1, static_cast<int>(roundf(static_cast<float>(input_width) * scale)));
    const int resized_height = max(1, static_cast<int>(roundf(static_cast<float>(input_height) * scale)));
    const bool inside =
        x >= pad_x && x < pad_x + resized_width &&
        y >= pad_y && y < pad_y + resized_height;

    float r = 114.0F;
    float g = 114.0F;
    float b = 114.0F;
    if (inside) {
        const float src_x = (static_cast<float>(x - pad_x) + 0.5F) / scale - 0.5F;
        const float src_y = (static_cast<float>(y - pad_y) + 0.5F) / scale - 0.5F;
        b = SampleBilinear(input_bgr, input_width, input_height, input_stride_bytes, src_x, src_y, 0);
        g = SampleBilinear(input_bgr, input_width, input_height, input_stride_bytes, src_x, src_y, 1);
        r = SampleBilinear(input_bgr, input_width, input_height, input_stride_bytes, src_x, src_y, 2);
    }

    const size_t batch_offset =
        static_cast<size_t>(batch_index) * 3U * static_cast<size_t>(plane_size);
    output_nchw[batch_offset + static_cast<size_t>(0) * plane_size + pixel_index] = r / 255.0F;
    output_nchw[batch_offset + static_cast<size_t>(1) * plane_size + pixel_index] = g / 255.0F;
    output_nchw[batch_offset + static_cast<size_t>(2) * plane_size + pixel_index] = b / 255.0F;
}

}  // namespace

cudaError_t LaunchPreprocessKernel(
    const uint8_t* input_bgr,
    int input_width,
    int input_height,
    int input_stride_bytes,
    float* output_nchw,
    int output_width,
    int output_height,
    float scale,
    int pad_x,
    int pad_y,
    int batch_index,
    int batch_size,
    cudaStream_t stream) {
    (void)batch_size;
    constexpr int threads = 256;
    const int pixels = output_width * output_height;
    const int blocks = (pixels + threads - 1) / threads;
    PreprocessKernel<<<blocks, threads, 0, stream>>>(
        input_bgr,
        input_width,
        input_height,
        input_stride_bytes,
        output_nchw,
        output_width,
        output_height,
        scale,
        pad_x,
        pad_y,
        batch_index,
        batch_size);
    return cudaGetLastError();
}

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA
