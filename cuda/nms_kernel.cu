#include "cuda/nms_kernel.h"

#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)

#include <cstddef>

#include <cuda_runtime.h>

namespace edge {
namespace {

constexpr int kFloatCandidateValues = 7;

__device__ float CandidateArea(const GpuDecodedCandidate& det) {
    const float w = fmaxf(0.0F, det.x2 - det.x1);
    const float h = fmaxf(0.0F, det.y2 - det.y1);
    return w * h;
}

__device__ float CandidateIoU(const GpuDecodedCandidate& lhs, const GpuDecodedCandidate& rhs) {
    const float xx1 = fmaxf(lhs.x1, rhs.x1);
    const float yy1 = fmaxf(lhs.y1, rhs.y1);
    const float xx2 = fminf(lhs.x2, rhs.x2);
    const float yy2 = fminf(lhs.y2, rhs.y2);
    const float inter_w = fmaxf(0.0F, xx2 - xx1);
    const float inter_h = fmaxf(0.0F, yy2 - yy1);
    const float inter = inter_w * inter_h;
    const float denom = CandidateArea(lhs) + CandidateArea(rhs) - inter;
    return denom > 0.0F ? inter / denom : 0.0F;
}

__device__ GpuDecodedCandidate LoadFloatCandidate(const float* candidates, int index) {
    const int base = index * kFloatCandidateValues;
    GpuDecodedCandidate det{};
    det.x1 = candidates[base + 0];
    det.y1 = candidates[base + 1];
    det.x2 = candidates[base + 2];
    det.y2 = candidates[base + 3];
    det.score = candidates[base + 4];
    det.class_id = static_cast<int>(roundf(candidates[base + 5]));
    det.batch_index = static_cast<int>(roundf(candidates[base + 6]));
    return det;
}

__device__ void StoreFloatCandidate(float* candidates, int index, const GpuDecodedCandidate& det) {
    const int base = index * kFloatCandidateValues;
    candidates[base + 0] = det.x1;
    candidates[base + 1] = det.y1;
    candidates[base + 2] = det.x2;
    candidates[base + 3] = det.y2;
    candidates[base + 4] = det.score;
    candidates[base + 5] = static_cast<float>(det.class_id);
    candidates[base + 6] = static_cast<float>(det.batch_index);
}

__global__ void NMSKernel(
    const GpuDecodedCandidate* candidates,
    const int* candidate_counts,
    int batch,
    int top_k,
    float nms_threshold,
    GpuDecodedCandidate* kept_candidates,
    int* kept_counts) {
    const int batch_index = blockIdx.x;
    if (batch_index >= batch || threadIdx.x != 0) {
        return;
    }

    extern __shared__ unsigned char suppressed[];
    int valid_count = candidate_counts[batch_index];
    valid_count = valid_count < 0 ? 0 : valid_count;
    valid_count = valid_count > top_k ? top_k : valid_count;
    for (int i = 0; i < valid_count; ++i) {
        suppressed[i] = 0;
    }

    int kept = 0;
    for (int step = 0; step < valid_count && kept < top_k; ++step) {
        int best_index = -1;
        float best_score = -1.0F;
        for (int i = 0; i < valid_count; ++i) {
            if (suppressed[i] != 0) {
                continue;
            }
            const GpuDecodedCandidate candidate = candidates[batch_index * top_k + i];
            if (candidate.score > best_score) {
                best_score = candidate.score;
                best_index = i;
            }
        }
        if (best_index < 0) {
            break;
        }

        const GpuDecodedCandidate selected = candidates[batch_index * top_k + best_index];
        kept_candidates[batch_index * top_k + kept] = selected;
        ++kept;
        suppressed[best_index] = 1;

        for (int i = 0; i < valid_count; ++i) {
            if (suppressed[i] != 0) {
                continue;
            }
            const GpuDecodedCandidate candidate = candidates[batch_index * top_k + i];
            if (candidate.class_id == selected.class_id &&
                CandidateIoU(candidate, selected) > nms_threshold) {
                suppressed[i] = 1;
            }
        }
    }
    kept_counts[batch_index] = kept;
}

__global__ void NMSFloatCandidatesKernel(
    const float* candidates,
    const int* candidate_counts,
    int batch,
    int top_k,
    float nms_threshold,
    float* kept_candidates,
    int* kept_counts) {
    const int batch_index = blockIdx.x;
    if (batch_index >= batch || threadIdx.x != 0) {
        return;
    }

    extern __shared__ unsigned char suppressed[];
    int valid_count = candidate_counts[batch_index];
    valid_count = valid_count < 0 ? 0 : valid_count;
    valid_count = valid_count > top_k ? top_k : valid_count;
    for (int i = 0; i < valid_count; ++i) {
        suppressed[i] = 0;
    }

    int kept = 0;
    for (int step = 0; step < valid_count && kept < top_k; ++step) {
        int best_index = -1;
        float best_score = -1.0F;
        for (int i = 0; i < valid_count; ++i) {
            if (suppressed[i] != 0) {
                continue;
            }
            const GpuDecodedCandidate candidate = LoadFloatCandidate(candidates, batch_index * top_k + i);
            if (candidate.score > best_score) {
                best_score = candidate.score;
                best_index = i;
            }
        }
        if (best_index < 0) {
            break;
        }

        const GpuDecodedCandidate selected = LoadFloatCandidate(candidates, batch_index * top_k + best_index);
        StoreFloatCandidate(kept_candidates, batch_index * top_k + kept, selected);
        ++kept;
        suppressed[best_index] = 1;

        for (int i = 0; i < valid_count; ++i) {
            if (suppressed[i] != 0) {
                continue;
            }
            const GpuDecodedCandidate candidate = LoadFloatCandidate(candidates, batch_index * top_k + i);
            if (candidate.class_id == selected.class_id &&
                CandidateIoU(candidate, selected) > nms_threshold) {
                suppressed[i] = 1;
            }
        }
    }
    kept_counts[batch_index] = kept;
}

}  // namespace

cudaError_t LaunchNMSKernel(
    const GpuDecodedCandidate* candidates,
    const int* candidate_counts,
    int batch,
    int top_k,
    float nms_threshold,
    GpuDecodedCandidate* kept_candidates,
    int* kept_counts,
    cudaStream_t stream) {
    if (batch <= 0 || top_k <= 0) {
        return cudaSuccess;
    }
    const std::size_t shared_bytes = static_cast<std::size_t>(top_k) * sizeof(unsigned char);
    NMSKernel<<<batch, 1, shared_bytes, stream>>>(
        candidates,
        candidate_counts,
        batch,
        top_k,
        nms_threshold,
        kept_candidates,
        kept_counts);
    return cudaGetLastError();
}

cudaError_t LaunchNMSFloatCandidatesKernel(
    const float* candidates,
    const int* candidate_counts,
    int batch,
    int top_k,
    float nms_threshold,
    float* kept_candidates,
    int* kept_counts,
    cudaStream_t stream) {
    if (batch <= 0 || top_k <= 0) {
        return cudaSuccess;
    }
    const std::size_t shared_bytes = static_cast<std::size_t>(top_k) * sizeof(unsigned char);
    NMSFloatCandidatesKernel<<<batch, 1, shared_bytes, stream>>>(
        candidates,
        candidate_counts,
        batch,
        top_k,
        nms_threshold,
        kept_candidates,
        kept_counts);
    return cudaGetLastError();
}

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA || EDGE_ENABLE_TENSORRT_PLUGIN
