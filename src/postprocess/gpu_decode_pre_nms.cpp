#include "postprocess/gpu_decode_pre_nms.h"

#ifdef EDGE_ENABLE_CUDA

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/logging.h"
#include "cuda/decode_kernel.h"
#include "cuda/cuda_runtime_utils.h"
#include "cuda/nms_kernel.h"
#include "profiling/nvtx_utils.h"

namespace edge {
namespace {

double ElapsedMs(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return static_cast<double>(us) / 1000.0;
}

bool CheckCuda(cudaError_t status, const std::string& what) {
    if (status == cudaSuccess) {
        return true;
    }
    EDGE_LOG_ERROR(what << " failed: " << cudaGetErrorString(status));
    return false;
}

std::size_t CountElements(const std::vector<int64_t>& shape) {
    std::size_t elements = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        elements *= static_cast<std::size_t>(dim);
    }
    return elements;
}

}  // namespace

GpuDecodePreNMS::GpuDecodePreNMS(PostprocessConfig config, bool use_pinned_memory)
    : config_(std::move(config)),
      use_pinned_memory_(use_pinned_memory) {}

bool GpuDecodePreNMS::Run(
    const TensorBuffer& model_output,
    const std::vector<FrameMeta>& frame_metas,
    const std::vector<PreprocessMeta>& preprocess_metas,
    int input_width,
    int input_height,
    cudaStream_t stream,
    std::vector<Detection>& decoded,
    double* gpu_decode_pre_nms_ms) const {
    const auto start = std::chrono::steady_clock::now();
    decoded.clear();

    if (model_output.shape.size() != 3 || model_output.shape[2] != 6) {
        EDGE_LOG_ERROR("GpuDecodePreNMS expects model output shape [batch, boxes, 6], got "
                       << ShapeToString(model_output.shape));
        return false;
    }

    const int batch = static_cast<int>(model_output.shape[0]);
    const int boxes = static_cast<int>(model_output.shape[1]);
    constexpr int values_per_box = 6;
    if (batch <= 0 || boxes <= 0) {
        return true;
    }
    if (static_cast<int>(frame_metas.size()) < batch || static_cast<int>(preprocess_metas.size()) < batch) {
        EDGE_LOG_ERROR("GpuDecodePreNMS got metadata size smaller than output batch");
        return false;
    }

    const std::size_t expected_elements =
        static_cast<std::size_t>(batch) * static_cast<std::size_t>(boxes) * values_per_box;
    if (model_output.NumElements() != expected_elements || CountElements(model_output.shape) != expected_elements) {
        EDGE_LOG_ERROR("GpuDecodePreNMS output tensor data size mismatch, expected "
                       << expected_elements << ", got " << model_output.NumElements());
        return false;
    }

    const int top_k = std::max(1, config_.top_k);
    const std::size_t model_bytes = expected_elements * sizeof(float);
    const std::size_t metas_bytes = static_cast<std::size_t>(batch) * sizeof(PreprocessMeta);
    const std::size_t candidates_count = static_cast<std::size_t>(batch) * static_cast<std::size_t>(top_k);
    const std::size_t candidates_bytes = candidates_count * sizeof(GpuDecodedCandidate);
    const std::size_t counts_bytes = static_cast<std::size_t>(batch) * sizeof(int);

    float* device_model_output = nullptr;
    PreprocessMeta* device_metas = nullptr;
    GpuDecodedCandidate* device_candidates = nullptr;
    int* device_counts = nullptr;

    auto cleanup = [&]() {
        cudaFree(device_model_output);
        cudaFree(device_metas);
        cudaFree(device_candidates);
        cudaFree(device_counts);
    };

    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_model_output), model_bytes),
                   "cudaMalloc decode model output") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_metas), metas_bytes),
                   "cudaMalloc decode preprocess metas") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_candidates), candidates_bytes),
                   "cudaMalloc decode candidates") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_counts), counts_bytes),
                   "cudaMalloc decode counts")) {
        cleanup();
        return false;
    }

    if (!CheckCuda(cudaMemcpyAsync(device_model_output,
                                   model_output.Data(),
                                   model_bytes,
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync decode model output") ||
        !CheckCuda(cudaMemcpyAsync(device_metas,
                                   preprocess_metas.data(),
                                   metas_bytes,
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync decode preprocess metas") ||
        !CheckCuda(cudaMemsetAsync(device_candidates, 0, candidates_bytes, stream),
                   "cudaMemsetAsync decode candidates") ||
        !CheckCuda(cudaMemsetAsync(device_counts, 0, counts_bytes, stream),
                   "cudaMemsetAsync decode counts")) {
        cleanup();
        return false;
    }

    if (!CheckCuda(LaunchDecodePreNMSKernel(device_model_output,
                                            batch,
                                            boxes,
                                            values_per_box,
                                            device_metas,
                                            config_.score_threshold,
                                            top_k,
                                            input_width,
                                            input_height,
                                            device_candidates,
                                            device_counts,
                                            stream),
                   "LaunchDecodePreNMSKernel")) {
        cleanup();
        return false;
    }

    std::vector<int> host_counts_vector;
    std::vector<GpuDecodedCandidate> host_candidates_vector;
    PinnedHostBuffer<int> pinned_counts;
    PinnedHostBuffer<GpuDecodedCandidate> pinned_candidates;
    int* host_counts = nullptr;
    GpuDecodedCandidate* host_candidates = nullptr;
    if (use_pinned_memory_) {
        if (!pinned_counts.Allocate(static_cast<std::size_t>(batch)) ||
            !pinned_candidates.Allocate(candidates_count)) {
            EDGE_LOG_ERROR("cudaHostAlloc failed for GPU decode pinned output buffers");
            cleanup();
            return false;
        }
        host_counts = pinned_counts.Data();
        host_candidates = pinned_candidates.Data();
    } else {
        host_counts_vector.assign(static_cast<std::size_t>(batch), 0);
        host_candidates_vector.resize(candidates_count);
        host_counts = host_counts_vector.data();
        host_candidates = host_candidates_vector.data();
    }
    if (!CheckCuda(cudaMemcpyAsync(host_counts,
                                   device_counts,
                                   counts_bytes,
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync decode counts to host") ||
        !CheckCuda(cudaMemcpyAsync(host_candidates,
                                   device_candidates,
                                   candidates_bytes,
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync decode candidates to host") ||
        !CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize gpu decode pre-nms")) {
        cleanup();
        return false;
    }
    cleanup();

    decoded.reserve(candidates_count);
    for (int b = 0; b < batch; ++b) {
        const int valid_count = std::min(host_counts[static_cast<std::size_t>(b)], top_k);
        for (int i = 0; i < valid_count; ++i) {
            const auto& candidate =
                host_candidates[static_cast<std::size_t>(b) * static_cast<std::size_t>(top_k) +
                                static_cast<std::size_t>(i)];
            Detection det;
            det.stream_id = frame_metas[static_cast<std::size_t>(candidate.batch_index)].stream_id;
            det.frame_id = frame_metas[static_cast<std::size_t>(candidate.batch_index)].frame_id;
            det.class_id = candidate.class_id;
            det.score = candidate.score;
            det.x1 = candidate.x1;
            det.y1 = candidate.y1;
            det.x2 = candidate.x2;
            det.y2 = candidate.y2;
            decoded.push_back(det);
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = ElapsedMs(start, end);
    if (gpu_decode_pre_nms_ms != nullptr) {
        *gpu_decode_pre_nms_ms = elapsed_ms;
    }
    EDGE_LOG_INFO("[Postprocess] gpu_decode_pre_nms_latency_ms="
                  << elapsed_ms << ", pinned_host=" << (use_pinned_memory_ ? "true" : "false")
                  << ", candidates=" << decoded.size()
                  << ", top_k_per_batch=" << top_k);
    return true;
}

bool GpuDecodePreNMS::RunWithGpuNms(
    const TensorBuffer& model_output,
    const std::vector<FrameMeta>& frame_metas,
    const std::vector<PreprocessMeta>& preprocess_metas,
    int input_width,
    int input_height,
    cudaStream_t stream,
    std::vector<Detection>& final_detections,
    double* gpu_decode_pre_nms_ms,
    double* gpu_nms_ms) const {
    const auto total_start = std::chrono::steady_clock::now();
    final_detections.clear();

    if (model_output.shape.size() != 3 || model_output.shape[2] != 6) {
        EDGE_LOG_ERROR("GpuDecodePreNMS::RunWithGpuNms expects model output shape [batch, boxes, 6], got "
                       << ShapeToString(model_output.shape));
        return false;
    }

    const int batch = static_cast<int>(model_output.shape[0]);
    const int boxes = static_cast<int>(model_output.shape[1]);
    constexpr int values_per_box = 6;
    if (batch <= 0 || boxes <= 0) {
        return true;
    }
    if (static_cast<int>(frame_metas.size()) < batch || static_cast<int>(preprocess_metas.size()) < batch) {
        EDGE_LOG_ERROR("GpuDecodePreNMS::RunWithGpuNms got metadata size smaller than output batch");
        return false;
    }

    const std::size_t expected_elements =
        static_cast<std::size_t>(batch) * static_cast<std::size_t>(boxes) * values_per_box;
    if (model_output.NumElements() != expected_elements || CountElements(model_output.shape) != expected_elements) {
        EDGE_LOG_ERROR("GpuDecodePreNMS::RunWithGpuNms output tensor data size mismatch, expected "
                       << expected_elements << ", got " << model_output.NumElements());
        return false;
    }

    const int top_k = std::max(1, config_.top_k);
    const std::size_t model_bytes = expected_elements * sizeof(float);
    const std::size_t metas_bytes = static_cast<std::size_t>(batch) * sizeof(PreprocessMeta);
    const std::size_t candidates_count = static_cast<std::size_t>(batch) * static_cast<std::size_t>(top_k);
    const std::size_t candidates_bytes = candidates_count * sizeof(GpuDecodedCandidate);
    const std::size_t counts_bytes = static_cast<std::size_t>(batch) * sizeof(int);

    float* device_model_output = nullptr;
    PreprocessMeta* device_metas = nullptr;
    GpuDecodedCandidate* device_candidates = nullptr;
    int* device_counts = nullptr;
    GpuDecodedCandidate* device_kept = nullptr;
    int* device_kept_counts = nullptr;

    auto cleanup = [&]() {
        cudaFree(device_model_output);
        cudaFree(device_metas);
        cudaFree(device_candidates);
        cudaFree(device_counts);
        cudaFree(device_kept);
        cudaFree(device_kept_counts);
    };

    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_model_output), model_bytes),
                   "cudaMalloc gpu nms model output") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_metas), metas_bytes),
                   "cudaMalloc gpu nms preprocess metas") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_candidates), candidates_bytes),
                   "cudaMalloc gpu nms candidates") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_counts), counts_bytes),
                   "cudaMalloc gpu nms counts") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_kept), candidates_bytes),
                   "cudaMalloc gpu nms kept candidates") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_kept_counts), counts_bytes),
                   "cudaMalloc gpu nms kept counts")) {
        cleanup();
        return false;
    }

    const auto decode_start = std::chrono::steady_clock::now();
    {
        PROFILE_RANGE("decode");
        if (!CheckCuda(cudaMemcpyAsync(device_model_output,
                                       model_output.Data(),
                                       model_bytes,
                                       cudaMemcpyHostToDevice,
                                       stream),
                       "cudaMemcpyAsync gpu nms model output") ||
            !CheckCuda(cudaMemcpyAsync(device_metas,
                                       preprocess_metas.data(),
                                       metas_bytes,
                                       cudaMemcpyHostToDevice,
                                       stream),
                       "cudaMemcpyAsync gpu nms preprocess metas") ||
            !CheckCuda(cudaMemsetAsync(device_candidates, 0, candidates_bytes, stream),
                       "cudaMemsetAsync gpu nms candidates") ||
            !CheckCuda(cudaMemsetAsync(device_counts, 0, counts_bytes, stream),
                       "cudaMemsetAsync gpu nms counts")) {
            cleanup();
            return false;
        }

        if (!CheckCuda(LaunchDecodePreNMSKernel(device_model_output,
                                                batch,
                                                boxes,
                                                values_per_box,
                                                device_metas,
                                                config_.score_threshold,
                                                top_k,
                                                input_width,
                                                input_height,
                                                device_candidates,
                                                device_counts,
                                                stream),
                       "LaunchDecodePreNMSKernel for gpu nms") ||
            !CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize decode before gpu nms")) {
            cleanup();
            return false;
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();

    const auto nms_start = std::chrono::steady_clock::now();
    std::vector<int> host_counts_vector;
    std::vector<GpuDecodedCandidate> host_kept_vector;
    PinnedHostBuffer<int> pinned_counts;
    PinnedHostBuffer<GpuDecodedCandidate> pinned_kept;
    int* host_counts = nullptr;
    GpuDecodedCandidate* host_kept = nullptr;
    if (use_pinned_memory_) {
        if (!pinned_counts.Allocate(static_cast<std::size_t>(batch)) ||
            !pinned_kept.Allocate(candidates_count)) {
            EDGE_LOG_ERROR("cudaHostAlloc failed for GPU NMS pinned output buffers");
            cleanup();
            return false;
        }
        host_counts = pinned_counts.Data();
        host_kept = pinned_kept.Data();
    } else {
        host_counts_vector.assign(static_cast<std::size_t>(batch), 0);
        host_kept_vector.resize(candidates_count);
        host_counts = host_counts_vector.data();
        host_kept = host_kept_vector.data();
    }
    {
        PROFILE_RANGE("nms");
        if (!CheckCuda(cudaMemsetAsync(device_kept, 0, candidates_bytes, stream),
                       "cudaMemsetAsync gpu nms kept candidates") ||
            !CheckCuda(cudaMemsetAsync(device_kept_counts, 0, counts_bytes, stream),
                       "cudaMemsetAsync gpu nms kept counts") ||
            !CheckCuda(LaunchNMSKernel(device_candidates,
                                       device_counts,
                                       batch,
                                       top_k,
                                       config_.nms_threshold,
                                       device_kept,
                                       device_kept_counts,
                                       stream),
                       "LaunchNMSKernel")) {
            cleanup();
            return false;
        }

        if (!CheckCuda(cudaMemcpyAsync(host_counts,
                                       device_kept_counts,
                                       counts_bytes,
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync gpu nms counts to host") ||
            !CheckCuda(cudaMemcpyAsync(host_kept,
                                       device_kept,
                                       candidates_bytes,
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync gpu nms candidates to host") ||
            !CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize gpu nms")) {
            cleanup();
            return false;
        }
    }
    const auto nms_end = std::chrono::steady_clock::now();
    cleanup();

    final_detections.reserve(candidates_count);
    for (int b = 0; b < batch; ++b) {
        const int valid_count = std::min(host_counts[static_cast<std::size_t>(b)], top_k);
        for (int i = 0; i < valid_count; ++i) {
            const auto& candidate =
                host_kept[static_cast<std::size_t>(b) * static_cast<std::size_t>(top_k) +
                          static_cast<std::size_t>(i)];
            Detection det;
            det.stream_id = frame_metas[static_cast<std::size_t>(candidate.batch_index)].stream_id;
            det.frame_id = frame_metas[static_cast<std::size_t>(candidate.batch_index)].frame_id;
            det.class_id = candidate.class_id;
            det.score = candidate.score;
            det.x1 = candidate.x1;
            det.y1 = candidate.y1;
            det.x2 = candidate.x2;
            det.y2 = candidate.y2;
            final_detections.push_back(det);
        }
    }

    const auto total_end = std::chrono::steady_clock::now();
    const double decode_ms = ElapsedMs(decode_start, decode_end);
    const double nms_ms = ElapsedMs(nms_start, nms_end);
    const double total_ms = ElapsedMs(total_start, total_end);
    if (gpu_decode_pre_nms_ms != nullptr) {
        *gpu_decode_pre_nms_ms = decode_ms;
    }
    if (gpu_nms_ms != nullptr) {
        *gpu_nms_ms = nms_ms;
    }
    EDGE_LOG_INFO("[Postprocess] gpu_decode_pre_nms_latency_ms="
                  << decode_ms << ", gpu_nms_latency_ms=" << nms_ms
                  << ", gpu_postprocess_latency_ms=" << total_ms
                  << ", pinned_host=" << (use_pinned_memory_ ? "true" : "false")
                  << ", detection_count=" << final_detections.size()
                  << ", top_k_per_batch=" << top_k);
    return true;
}

std::string GpuDecodePreNMS::Name() const {
    return "GpuDecodePreNMS";
}

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA
