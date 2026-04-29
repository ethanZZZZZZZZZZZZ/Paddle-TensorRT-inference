#pragma once

#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)

#include <cuda_runtime.h>

#include "common/types.h"

namespace edge {

struct GpuDecodedCandidate {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
    int batch_index;
};

cudaError_t LaunchDecodePreNMSKernel(
    const float* model_output,
    int batch,
    int boxes,
    int values_per_box,
    const PreprocessMeta* preprocess_metas,
    float score_threshold,
    int top_k,
    int input_width,
    int input_height,
    GpuDecodedCandidate* candidates,
    int* candidate_counts,
    cudaStream_t stream);

cudaError_t LaunchDecodePreNMSFloatMetaKernel(
    const float* model_output,
    int batch,
    int boxes,
    int values_per_box,
    const float* preprocess_metas,
    float score_threshold,
    int top_k,
    int input_width,
    int input_height,
    float* candidates,
    int* candidate_counts,
    cudaStream_t stream);

cudaError_t LaunchDecodePreNMSHalfMetaKernel(
    const void* model_output,
    int batch,
    int boxes,
    int values_per_box,
    const float* preprocess_metas,
    float score_threshold,
    int top_k,
    int input_width,
    int input_height,
    float* candidates,
    int* candidate_counts,
    cudaStream_t stream);

cudaError_t LaunchDecodePreNMSInt8MetaKernel(
    const void* model_output,
    float input_scale,
    int batch,
    int boxes,
    int values_per_box,
    const float* preprocess_metas,
    float score_threshold,
    int top_k,
    int input_width,
    int input_height,
    float* candidates,
    int* candidate_counts,
    cudaStream_t stream);

cudaError_t LaunchDecodePreNMSBcnFloatMetaKernel(
    const float* model_output,
    int batch,
    int channels,
    int boxes,
    const float* preprocess_metas,
    float score_threshold,
    int top_k,
    int input_width,
    int input_height,
    float* candidates,
    int* candidate_counts,
    cudaStream_t stream);

cudaError_t LaunchDecodePreNMSBcnHalfMetaKernel(
    const void* model_output,
    int batch,
    int channels,
    int boxes,
    const float* preprocess_metas,
    float score_threshold,
    int top_k,
    int input_width,
    int input_height,
    float* candidates,
    int* candidate_counts,
    cudaStream_t stream);

cudaError_t LaunchDecodePreNMSBcnInt8MetaKernel(
    const void* model_output,
    float input_scale,
    int batch,
    int channels,
    int boxes,
    const float* preprocess_metas,
    float score_threshold,
    int top_k,
    int input_width,
    int input_height,
    float* candidates,
    int* candidate_counts,
    cudaStream_t stream);

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA || EDGE_ENABLE_TENSORRT_PLUGIN
