#include "cuda/decode_kernel.h"

#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)

#include <cuda_runtime.h>

namespace edge {
namespace {

__device__ float ClampFloat(float value, float low, float high) {
    return fminf(fmaxf(value, low), high);
}

__device__ void SortBoxCorners(float* x1, float* y1, float* x2, float* y2) {
    if (*x2 < *x1) {
        const float tmp = *x1;
        *x1 = *x2;
        *x2 = tmp;
    }
    if (*y2 < *y1) {
        const float tmp = *y1;
        *y1 = *y2;
        *y2 = tmp;
    }
}

__global__ void DecodePreNMSKernel(
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
    int* candidate_counts) {
    (void)input_width;
    (void)input_height;

    const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_boxes = batch * boxes;
    if (linear_index >= total_boxes) {
        return;
    }

    const int batch_index = linear_index / boxes;
    const int box_index = linear_index % boxes;
    const int base = (batch_index * boxes + box_index) * values_per_box;
    const float score = model_output[base + 4];
    if (score < score_threshold) {
        return;
    }

    const int slot = atomicAdd(candidate_counts + batch_index, 1);
    if (slot >= top_k) {
        return;
    }

    const PreprocessMeta meta = preprocess_metas[batch_index];
    const float scale = meta.scale > 0.0F ? meta.scale : 1.0F;
    const float max_x = static_cast<float>(meta.original_width - 1);
    const float max_y = static_cast<float>(meta.original_height - 1);

    float x1 = (model_output[base + 0] - static_cast<float>(meta.pad_x)) / scale;
    float y1 = (model_output[base + 1] - static_cast<float>(meta.pad_y)) / scale;
    float x2 = (model_output[base + 2] - static_cast<float>(meta.pad_x)) / scale;
    float y2 = (model_output[base + 3] - static_cast<float>(meta.pad_y)) / scale;

    x1 = ClampFloat(x1, 0.0F, max_x);
    y1 = ClampFloat(y1, 0.0F, max_y);
    x2 = ClampFloat(x2, 0.0F, max_x);
    y2 = ClampFloat(y2, 0.0F, max_y);

    SortBoxCorners(&x1, &y1, &x2, &y2);

    const int candidate_index = batch_index * top_k + slot;
    candidates[candidate_index].x1 = x1;
    candidates[candidate_index].y1 = y1;
    candidates[candidate_index].x2 = x2;
    candidates[candidate_index].y2 = y2;
    candidates[candidate_index].score = score;
    candidates[candidate_index].class_id = static_cast<int>(roundf(model_output[base + 5]));
    candidates[candidate_index].batch_index = batch_index;
}

__global__ void DecodePreNMSFloatMetaKernel(
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
    int* candidate_counts) {
    (void)input_width;
    (void)input_height;

    const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_boxes = batch * boxes;
    if (linear_index >= total_boxes) {
        return;
    }

    const int batch_index = linear_index / boxes;
    const int box_index = linear_index % boxes;
    const int base = (batch_index * boxes + box_index) * values_per_box;
    const float score = model_output[base + 4];
    if (score < score_threshold) {
        return;
    }

    const int slot = atomicAdd(candidate_counts + batch_index, 1);
    if (slot >= top_k) {
        return;
    }

    const int meta_base = batch_index * 7;
    const float original_width = preprocess_metas[meta_base + 0];
    const float original_height = preprocess_metas[meta_base + 1];
    const float scale_value = preprocess_metas[meta_base + 4];
    const float pad_x = preprocess_metas[meta_base + 5];
    const float pad_y = preprocess_metas[meta_base + 6];
    const float scale = scale_value > 0.0F ? scale_value : 1.0F;
    const float max_x = fmaxf(original_width - 1.0F, 0.0F);
    const float max_y = fmaxf(original_height - 1.0F, 0.0F);

    float x1 = (model_output[base + 0] - pad_x) / scale;
    float y1 = (model_output[base + 1] - pad_y) / scale;
    float x2 = (model_output[base + 2] - pad_x) / scale;
    float y2 = (model_output[base + 3] - pad_y) / scale;

    x1 = ClampFloat(x1, 0.0F, max_x);
    y1 = ClampFloat(y1, 0.0F, max_y);
    x2 = ClampFloat(x2, 0.0F, max_x);
    y2 = ClampFloat(y2, 0.0F, max_y);
    SortBoxCorners(&x1, &y1, &x2, &y2);

    const int candidate_base = (batch_index * top_k + slot) * 7;
    candidates[candidate_base + 0] = x1;
    candidates[candidate_base + 1] = y1;
    candidates[candidate_base + 2] = x2;
    candidates[candidate_base + 3] = y2;
    candidates[candidate_base + 4] = score;
    candidates[candidate_base + 5] = roundf(model_output[base + 5]);
    candidates[candidate_base + 6] = static_cast<float>(batch_index);
}

__global__ void DecodePreNMSBcnFloatMetaKernel(
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
    int* candidate_counts) {
    (void)input_width;
    (void)input_height;

    const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_boxes = batch * boxes;
    if (linear_index >= total_boxes || channels < 5) {
        return;
    }

    const int batch_index = linear_index / boxes;
    const int box_index = linear_index % boxes;
    const int tensor_base = batch_index * channels * boxes;

    float best_score = model_output[tensor_base + 4 * boxes + box_index];
    int best_class = 0;
    for (int c = 5; c < channels; ++c) {
        const float score = model_output[tensor_base + c * boxes + box_index];
        if (score > best_score) {
            best_score = score;
            best_class = c - 4;
        }
    }
    if (best_score < score_threshold) {
        return;
    }

    const int slot = atomicAdd(candidate_counts + batch_index, 1);
    if (slot >= top_k) {
        return;
    }

    const int meta_base = batch_index * 7;
    const float original_width = preprocess_metas[meta_base + 0];
    const float original_height = preprocess_metas[meta_base + 1];
    const float scale_value = preprocess_metas[meta_base + 4];
    const float pad_x = preprocess_metas[meta_base + 5];
    const float pad_y = preprocess_metas[meta_base + 6];
    const float scale = scale_value > 0.0F ? scale_value : 1.0F;
    const float max_x = fmaxf(original_width - 1.0F, 0.0F);
    const float max_y = fmaxf(original_height - 1.0F, 0.0F);

    const float cx = model_output[tensor_base + box_index];
    const float cy = model_output[tensor_base + boxes + box_index];
    const float w = model_output[tensor_base + 2 * boxes + box_index];
    const float h = model_output[tensor_base + 3 * boxes + box_index];

    float x1 = (cx - 0.5F * w - pad_x) / scale;
    float y1 = (cy - 0.5F * h - pad_y) / scale;
    float x2 = (cx + 0.5F * w - pad_x) / scale;
    float y2 = (cy + 0.5F * h - pad_y) / scale;

    x1 = ClampFloat(x1, 0.0F, max_x);
    y1 = ClampFloat(y1, 0.0F, max_y);
    x2 = ClampFloat(x2, 0.0F, max_x);
    y2 = ClampFloat(y2, 0.0F, max_y);
    SortBoxCorners(&x1, &y1, &x2, &y2);

    const int candidate_base = (batch_index * top_k + slot) * 7;
    candidates[candidate_base + 0] = x1;
    candidates[candidate_base + 1] = y1;
    candidates[candidate_base + 2] = x2;
    candidates[candidate_base + 3] = y2;
    candidates[candidate_base + 4] = best_score;
    candidates[candidate_base + 5] = static_cast<float>(best_class);
    candidates[candidate_base + 6] = static_cast<float>(batch_index);
}

__global__ void CapCandidateCountsKernel(int* candidate_counts, int batch, int top_k) {
    const int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_index >= batch) {
        return;
    }
    if (candidate_counts[batch_index] > top_k) {
        candidate_counts[batch_index] = top_k;
    }
}

}  // namespace

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
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int total_boxes = batch * boxes;
    const int blocks = (total_boxes + threads - 1) / threads;
    DecodePreNMSKernel<<<blocks, threads, 0, stream>>>(
        model_output,
        batch,
        boxes,
        values_per_box,
        preprocess_metas,
        score_threshold,
        top_k,
        input_width,
        input_height,
        candidates,
        candidate_counts);
    return cudaGetLastError();
}

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
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int total_boxes = batch * boxes;
    const int blocks = (total_boxes + threads - 1) / threads;
    DecodePreNMSFloatMetaKernel<<<blocks, threads, 0, stream>>>(
        model_output,
        batch,
        boxes,
        values_per_box,
        preprocess_metas,
        score_threshold,
        top_k,
        input_width,
        input_height,
        candidates,
        candidate_counts);

    const int count_blocks = (batch + threads - 1) / threads;
    CapCandidateCountsKernel<<<count_blocks, threads, 0, stream>>>(candidate_counts, batch, top_k);
    return cudaGetLastError();
}

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
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int total_boxes = batch * boxes;
    const int blocks = (total_boxes + threads - 1) / threads;
    DecodePreNMSBcnFloatMetaKernel<<<blocks, threads, 0, stream>>>(
        model_output,
        batch,
        channels,
        boxes,
        preprocess_metas,
        score_threshold,
        top_k,
        input_width,
        input_height,
        candidates,
        candidate_counts);

    const int count_blocks = (batch + threads - 1) / threads;
    CapCandidateCountsKernel<<<count_blocks, threads, 0, stream>>>(candidate_counts, batch, top_k);
    return cudaGetLastError();
}

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA || EDGE_ENABLE_TENSORRT_PLUGIN
