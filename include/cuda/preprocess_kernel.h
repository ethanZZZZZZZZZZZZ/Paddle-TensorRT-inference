#pragma once

#ifdef EDGE_ENABLE_CUDA

#include <cstdint>

#include <cuda_runtime.h>

namespace edge {

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
    cudaStream_t stream);

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA
