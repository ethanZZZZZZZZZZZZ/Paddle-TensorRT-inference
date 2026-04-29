#include "preprocess/preprocessor.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>

#ifdef EDGE_ENABLE_OPENCV
#include <opencv2/imgproc.hpp>
#endif

#include "common/logging.h"

namespace edge {
namespace {

double ElapsedMs(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return static_cast<double>(us) / 1000.0;
}

PreprocessMeta MakeMeta(const VideoFrame& frame, int input_width, int input_height) {
    const float scale_x = static_cast<float>(input_width) / static_cast<float>(frame.meta.width);
    const float scale_y = static_cast<float>(input_height) / static_cast<float>(frame.meta.height);
    const float scale = std::min(scale_x, scale_y);
    const int resized_w = std::max(1, static_cast<int>(std::round(static_cast<float>(frame.meta.width) * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(static_cast<float>(frame.meta.height) * scale)));

    PreprocessMeta meta;
    meta.original_width = frame.meta.width;
    meta.original_height = frame.meta.height;
    meta.input_width = input_width;
    meta.input_height = input_height;
    meta.scale = scale;
    meta.pad_x = (input_width - resized_w) / 2;
    meta.pad_y = (input_height - resized_h) / 2;
    return meta;
}

}  // namespace

CPUPreprocessor::CPUPreprocessor(int input_width, int input_height)
    : input_width_(input_width),
      input_height_(input_height) {}

bool CPUPreprocessor::Run(const VideoFrame& frame, TensorBuffer& output, PreprocessMeta& meta) const {
    std::vector<PreprocessMeta> metas;
    const std::vector<VideoFrame> frames{frame};
    if (!RunBatch(frames, output, metas)) {
        return false;
    }
    if (metas.empty()) {
        EDGE_LOG_ERROR("CPUPreprocessor produced no metadata for single-frame input");
        return false;
    }
    meta = metas.front();
    return true;
}

bool CPUPreprocessor::RunBatch(
    const std::vector<VideoFrame>& frames,
    TensorBuffer& output,
    std::vector<PreprocessMeta>& metas) const {
    const auto start = std::chrono::steady_clock::now();

    if (frames.empty()) {
        EDGE_LOG_ERROR("CPUPreprocessor received an empty batch");
        return false;
    }
    if (input_width_ <= 0 || input_height_ <= 0) {
        EDGE_LOG_ERROR("Invalid CPUPreprocessor input size");
        return false;
    }

    constexpr int channels = 3;
    output.shape = {static_cast<int64_t>(frames.size()), channels, input_height_, input_width_};
    output.ClearExternalHostData();
    output.host_data.assign(
        frames.size() * static_cast<size_t>(channels) * static_cast<size_t>(input_height_) *
            static_cast<size_t>(input_width_),
        0.0F);

    metas.clear();
    metas.reserve(frames.size());

    for (size_t batch_index = 0; batch_index < frames.size(); ++batch_index) {
        PreprocessMeta meta;
        if (!FillOne(frames[batch_index], batch_index, output, meta)) {
            return false;
        }
        metas.push_back(meta);
    }

    const auto end = std::chrono::steady_clock::now();
    EDGE_LOG_INFO("[Preprocess] cpu_preprocess_latency_ms=" << ElapsedMs(start, end)
                                                            << ", batch=" << frames.size()
                                                            << ", output_shape="
                                                            << ShapeToString(output.shape));
    return true;
}

bool CPUPreprocessor::FillOne(
    const VideoFrame& frame,
    std::size_t batch_index,
    TensorBuffer& output,
    PreprocessMeta& meta) const {
    if (frame.Empty()) {
        EDGE_LOG_ERROR("CPUPreprocessor received an empty frame");
        return false;
    }

    constexpr int channels = 3;
    if (frame.channels != channels) {
        EDGE_LOG_ERROR("CPUPreprocessor expects 3-channel BGR input, got " << frame.channels);
        return false;
    }

    meta = MakeMeta(frame, input_width_, input_height_);
    const int resized_w = std::max(1, static_cast<int>(std::round(static_cast<float>(frame.meta.width) * meta.scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(static_cast<float>(frame.meta.height) * meta.scale)));
    const size_t batch_offset =
        batch_index * static_cast<size_t>(channels) * static_cast<size_t>(input_height_) *
        static_cast<size_t>(input_width_);

#ifdef EDGE_ENABLE_OPENCV
    if (frame.image.empty()) {
        EDGE_LOG_ERROR("CPUPreprocessor received an empty cv::Mat");
        return false;
    }
    if (frame.image.channels() != channels) {
        EDGE_LOG_ERROR("CPUPreprocessor expects BGR cv::Mat with exactly 3 channels");
        return false;
    }

    cv::Mat resized;
    cv::resize(frame.image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat letterbox(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    const cv::Rect roi(meta.pad_x, meta.pad_y, resized_w, resized_h);
    resized.copyTo(letterbox(roi));

    cv::Mat rgb;
    cv::cvtColor(letterbox, rgb, cv::COLOR_BGR2RGB);

    for (int y = 0; y < input_height_; ++y) {
        const auto* row = rgb.ptr<uint8_t>(y);
        for (int x = 0; x < input_width_; ++x) {
            const size_t pixel_base = static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c) {
                const size_t dst_index =
                    batch_offset +
                    static_cast<size_t>(c) * static_cast<size_t>(input_height_) *
                        static_cast<size_t>(input_width_) +
                    static_cast<size_t>(y) * static_cast<size_t>(input_width_) +
                    static_cast<size_t>(x);
                output.host_data[dst_index] = static_cast<float>(row[pixel_base + static_cast<size_t>(c)]) / 255.0F;
            }
        }
    }
#else
    for (int y = 0; y < input_height_; ++y) {
        for (int x = 0; x < input_width_; ++x) {
            uint8_t rgb[channels] = {114, 114, 114};
            if (x >= meta.pad_x && x < meta.pad_x + resized_w &&
                y >= meta.pad_y && y < meta.pad_y + resized_h) {
                const int resized_x = x - meta.pad_x;
                const int resized_y = y - meta.pad_y;
                const int src_x = std::clamp(
                    static_cast<int>(static_cast<float>(resized_x) / meta.scale),
                    0,
                    frame.meta.width - 1);
                const int src_y = std::clamp(
                    static_cast<int>(static_cast<float>(resized_y) / meta.scale),
                    0,
                    frame.meta.height - 1);
                const size_t src_index =
                    (static_cast<size_t>(src_y) * static_cast<size_t>(frame.meta.width) +
                     static_cast<size_t>(src_x)) *
                    static_cast<size_t>(frame.channels);
                rgb[0] = frame.image[src_index + 2];
                rgb[1] = frame.image[src_index + 1];
                rgb[2] = frame.image[src_index + 0];
            }
            for (int c = 0; c < channels; ++c) {
                const size_t dst_index =
                    batch_offset +
                    static_cast<size_t>(c) * static_cast<size_t>(input_height_) *
                        static_cast<size_t>(input_width_) +
                    static_cast<size_t>(y) * static_cast<size_t>(input_width_) +
                    static_cast<size_t>(x);
                output.host_data[dst_index] = static_cast<float>(rgb[c]) / 255.0F;
            }
        }
    }
#endif

    EDGE_LOG_INFO("CPUPreprocessor sample stream_id=" << frame.meta.stream_id
                                                      << ", frame_id=" << frame.meta.frame_id
                                                      << ", original=" << meta.original_width
                                                      << "x" << meta.original_height
                                                      << ", input=" << meta.input_width
                                                      << "x" << meta.input_height
                                                      << ", scale=" << meta.scale
                                                      << ", pad_x=" << meta.pad_x
                                                      << ", pad_y=" << meta.pad_y);
    return true;
}

std::string CPUPreprocessor::Name() const {
    return "CPUPreprocessor";
}

}  // namespace edge
