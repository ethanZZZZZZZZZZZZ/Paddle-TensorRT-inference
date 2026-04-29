#include "postprocess/postprocessor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "common/logging.h"
#include "profiling/nvtx_utils.h"

namespace edge {
namespace {

float Area(const Detection& det) {
    const float w = std::max(0.0F, det.x2 - det.x1);
    const float h = std::max(0.0F, det.y2 - det.y1);
    return w * h;
}

float IoU(const Detection& lhs, const Detection& rhs) {
    const float xx1 = std::max(lhs.x1, rhs.x1);
    const float yy1 = std::max(lhs.y1, rhs.y1);
    const float xx2 = std::min(lhs.x2, rhs.x2);
    const float yy2 = std::min(lhs.y2, rhs.y2);
    const float inter_w = std::max(0.0F, xx2 - xx1);
    const float inter_h = std::max(0.0F, yy2 - yy1);
    const float inter = inter_w * inter_h;
    const float denom = Area(lhs) + Area(rhs) - inter;
    if (denom <= 0.0F) {
        return 0.0F;
    }
    return inter / denom;
}

double ElapsedMs(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return static_cast<double>(us) / 1000.0;
}

}  // namespace

CPUPostprocessor::CPUPostprocessor(PostprocessConfig config)
    : config_(config) {}

bool CPUPostprocessor::Run(
    const TensorBuffer& model_output,
    const std::vector<FrameMeta>& frame_metas,
    const std::vector<PreprocessMeta>& preprocess_metas,
    std::vector<Detection>& final_detections,
    PostprocessTiming* timing) const {
    PROFILE_RANGE("postprocess");
    final_detections.clear();
    const auto total_start = std::chrono::steady_clock::now();

    if (config_.mode == "raw") {
        const auto total_end = std::chrono::steady_clock::now();
        const double total_ms = ElapsedMs(total_start, total_end);
        if (timing != nullptr) {
            timing->decode_ms = 0.0;
            timing->cpu_decode_ms = 0.0;
            timing->gpu_decode_pre_nms_ms = 0.0;
            timing->nms_ms = 0.0;
            timing->gpu_nms_ms = 0.0;
            timing->trt_plugin_ms = 0.0;
            timing->total_ms = total_ms;
        }
        EDGE_LOG_WARN("CPUPostprocessor mode=raw: skip detection decode for real Paddle output. "
                      << "model_output_shape=" << ShapeToString(model_output.shape)
                      << ", output_elements=" << model_output.NumElements()
                      << ", detection_count=0");
        return true;
    }

    const auto decode_start = std::chrono::steady_clock::now();
    std::vector<Detection> decoded;
    {
        PROFILE_RANGE("decode");
        if (!DecodePreNms(model_output, frame_metas, preprocess_metas, decoded)) {
            return false;
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();

    const auto nms_start = std::chrono::steady_clock::now();
    {
        PROFILE_RANGE("nms");
        final_detections = Nms(std::move(decoded), config_.nms_threshold, config_.top_k);
    }
    const auto nms_end = std::chrono::steady_clock::now();
    const auto total_end = std::chrono::steady_clock::now();

    const double decode_ms = ElapsedMs(decode_start, decode_end);
    const double nms_ms = ElapsedMs(nms_start, nms_end);
    const double total_ms = ElapsedMs(total_start, total_end);
    if (timing != nullptr) {
        timing->decode_ms = decode_ms;
        timing->cpu_decode_ms = decode_ms;
        timing->gpu_decode_pre_nms_ms = 0.0;
        timing->nms_ms = nms_ms;
        timing->gpu_nms_ms = 0.0;
        timing->trt_plugin_ms = 0.0;
        timing->total_ms = total_ms;
    }

    EDGE_LOG_INFO("[Postprocess] decode_latency_ms=" << decode_ms
                                                     << ", decode_backend=cpu"
                                                     << ", nms_latency_ms=" << nms_ms
                                                     << ", postprocess_latency_ms=" << total_ms
                                                     << ", detection_count="
                                                     << final_detections.size());
    return true;
}

std::string CPUPostprocessor::Name() const {
    return "CPUPostprocessor";
}

bool CPUPostprocessor::DecodePreNms(
    const TensorBuffer& model_output,
    const std::vector<FrameMeta>& frame_metas,
    const std::vector<PreprocessMeta>& preprocess_metas,
    std::vector<Detection>& decoded) const {
    decoded.clear();

    if (model_output.shape.size() != 3 || model_output.shape[2] != 6) {
        EDGE_LOG_ERROR("CPUPostprocessor expects model output shape [batch, boxes, 6], got "
                       << ShapeToString(model_output.shape)
                       << ". For a real Paddle model whose output decoder is not implemented yet, "
                       << "set postprocess.mode: raw to validate the inference path only.");
        return false;
    }

    const int batch = static_cast<int>(model_output.shape[0]);
    const int boxes = static_cast<int>(model_output.shape[1]);
    constexpr int values_per_box = 6;
    const size_t expected_elements =
        static_cast<size_t>(std::max(0, batch)) * static_cast<size_t>(std::max(0, boxes)) * values_per_box;
    if (model_output.NumElements() != expected_elements) {
        EDGE_LOG_ERROR("CPUPostprocessor output tensor data size mismatch, expected "
                       << expected_elements << ", got " << model_output.NumElements());
        return false;
    }
    if (static_cast<int>(frame_metas.size()) < batch || static_cast<int>(preprocess_metas.size()) < batch) {
        EDGE_LOG_ERROR("CPUPostprocessor got metadata size smaller than output batch");
        return false;
    }

    if (batch <= 0 || boxes <= 0) {
        return true;
    }

    const int max_candidates_per_batch = std::max(1, config_.top_k);
    std::vector<int> kept_per_batch(static_cast<size_t>(batch), 0);
    decoded.reserve(static_cast<size_t>(batch) *
                    static_cast<size_t>(std::min(boxes, max_candidates_per_batch)));

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < boxes; ++i) {
            const size_t base =
                (static_cast<size_t>(b) * static_cast<size_t>(boxes) + static_cast<size_t>(i)) *
                values_per_box;
            Detection raw;
            const float* output_data = model_output.Data();
            raw.x1 = output_data[base + 0];
            raw.y1 = output_data[base + 1];
            raw.x2 = output_data[base + 2];
            raw.y2 = output_data[base + 3];
            raw.score = output_data[base + 4];
            raw.class_id = static_cast<int>(std::round(output_data[base + 5]));

            if (raw.score < config_.score_threshold) {
                continue;
            }
            if (kept_per_batch[static_cast<size_t>(b)] >= max_candidates_per_batch) {
                continue;
            }

            decoded.push_back(MapBoxToOriginal(raw, frame_metas[static_cast<size_t>(b)],
                                               preprocess_metas[static_cast<size_t>(b)]));
            ++kept_per_batch[static_cast<size_t>(b)];
        }
    }

    return true;
}

Detection CPUPostprocessor::MapBoxToOriginal(
    const Detection& input_space_detection,
    const FrameMeta& frame_meta,
    const PreprocessMeta& preprocess_meta) {
    Detection out = input_space_detection;
    const float scale = preprocess_meta.scale > 0.0F ? preprocess_meta.scale : 1.0F;

    out.x1 = (input_space_detection.x1 - static_cast<float>(preprocess_meta.pad_x)) / scale;
    out.y1 = (input_space_detection.y1 - static_cast<float>(preprocess_meta.pad_y)) / scale;
    out.x2 = (input_space_detection.x2 - static_cast<float>(preprocess_meta.pad_x)) / scale;
    out.y2 = (input_space_detection.y2 - static_cast<float>(preprocess_meta.pad_y)) / scale;

    out.x1 = std::clamp(out.x1, 0.0F, static_cast<float>(frame_meta.width - 1));
    out.y1 = std::clamp(out.y1, 0.0F, static_cast<float>(frame_meta.height - 1));
    out.x2 = std::clamp(out.x2, 0.0F, static_cast<float>(frame_meta.width - 1));
    out.y2 = std::clamp(out.y2, 0.0F, static_cast<float>(frame_meta.height - 1));

    if (out.x2 < out.x1) {
        std::swap(out.x1, out.x2);
    }
    if (out.y2 < out.y1) {
        std::swap(out.y1, out.y2);
    }

    out.stream_id = frame_meta.stream_id;
    out.frame_id = frame_meta.frame_id;
    return out;
}

std::vector<Detection> CPUPostprocessor::Nms(
    std::vector<Detection> detections,
    float nms_threshold,
    int top_k) {
    std::sort(detections.begin(), detections.end(), [](const Detection& lhs, const Detection& rhs) {
        return lhs.score > rhs.score;
    });

    std::vector<Detection> kept;
    kept.reserve(detections.size());

    for (const auto& candidate : detections) {
        bool suppressed = false;
        for (const auto& selected : kept) {
            const bool same_scope = candidate.stream_id == selected.stream_id &&
                                    candidate.frame_id == selected.frame_id &&
                                    candidate.class_id == selected.class_id;
            if (same_scope && IoU(candidate, selected) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            if (top_k > 0 && static_cast<int>(kept.size()) >= top_k) {
                break;
            }
        }
    }

    return kept;
}

}  // namespace edge
