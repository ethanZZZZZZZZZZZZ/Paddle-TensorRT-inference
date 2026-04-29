#pragma once

#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)

#include <cuda_runtime.h>

#include "cuda/decode_kernel.h"

namespace edge {

cudaError_t LaunchNMSKernel(
    const GpuDecodedCandidate* candidates,
    const int* candidate_counts,
    int batch,
    int top_k,
    float nms_threshold,
    GpuDecodedCandidate* kept_candidates,
    int* kept_counts,
    cudaStream_t stream);

cudaError_t LaunchNMSFloatCandidatesKernel(
    const float* candidates,
    const int* candidate_counts,
    int batch,
    int top_k,
    float nms_threshold,
    float* kept_candidates,
    int* kept_counts,
    cudaStream_t stream);

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA || EDGE_ENABLE_TENSORRT_PLUGIN
