#include "pipeline/video_source.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include "common/logging.h"

#ifdef EDGE_ENABLE_OPENCV
#include <opencv2/imgcodecs.hpp>
#endif

namespace edge {
namespace {

std::string Trim(const std::string& input) {
    const auto begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

#ifdef EDGE_ENABLE_OPENCV
bool LoadImageList(const std::string& list_path, std::vector<std::string>& image_paths) {
    image_paths.clear();

    std::ifstream input(list_path);
    if (!input.is_open()) {
        EDGE_LOG_ERROR("Failed to open image list: " << list_path);
        return false;
    }

    const std::filesystem::path base_dir = std::filesystem::path(list_path).parent_path();
    std::string line;
    int line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        line = Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::filesystem::path image_path(line);
        if (image_path.is_relative()) {
            std::error_code direct_ec;
            if (!std::filesystem::is_regular_file(image_path, direct_ec)) {
                image_path = base_dir / image_path;
            }
        }

        std::error_code ec;
        if (!std::filesystem::is_regular_file(image_path, ec)) {
            EDGE_LOG_ERROR("Invalid image path in " << list_path << ":" << line_no
                                                    << " path=" << image_path.string());
            return false;
        }
        image_paths.push_back(image_path.lexically_normal().string());
    }

    if (image_paths.empty()) {
        EDGE_LOG_ERROR("Image list is empty: " << list_path);
        return false;
    }
    return true;
}
#endif

}  // namespace

VideoSource::VideoSource(
    int stream_id,
    std::string source_type,
    std::string path,
    int width,
    int height,
    int channels,
    int total_frames)
    : stream_id_(stream_id),
      source_type_(std::move(source_type)),
      path_(std::move(path)),
      width_(width),
      height_(height),
      channels_(channels),
      total_frames_(total_frames) {}

bool VideoSource::Open() {
    if (path_.empty()) {
        EDGE_LOG_ERROR("VideoSource path must not be empty");
        return false;
    }

    next_frame_id_ = 0;
    frames_read_ = 0;
    read_start_time_ = std::chrono::steady_clock::now();

    if (source_type_ == "synthetic") {
        if (width_ <= 0 || height_ <= 0 || channels_ <= 0 || total_frames_ <= 0) {
            EDGE_LOG_ERROR("Invalid synthetic video source config");
            return false;
        }
        opened_ = true;
        EDGE_LOG_INFO(Name() << " opened with synthetic frames "
                             << width_ << "x" << height_ << "x" << channels_
                             << ", total_frames=" << total_frames_
                             << ", path=" << path_);
        return true;
    }

    if (source_type_ == "video_file") {
#ifdef EDGE_ENABLE_OPENCV
        if (!capture_.open(path_)) {
            EDGE_LOG_ERROR(Name() << " failed to open video file: " << path_);
            return false;
        }

        width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
        video_fps_hint_ = capture_.get(cv::CAP_PROP_FPS);
        channels_ = 3;
        if (width_ <= 0 || height_ <= 0) {
            EDGE_LOG_ERROR(Name() << " opened video but got invalid frame size");
            return false;
        }
        opened_ = true;
        EDGE_LOG_INFO(Name() << " opened video_file path=" << path_
                             << ", frame_size=" << width_ << "x" << height_
                             << ", fps_hint=" << video_fps_hint_
                             << ", max_frames=" << total_frames_);
        return true;
#else
        EDGE_LOG_ERROR(Name() << " requires OpenCV. Reconfigure with -DENABLE_OPENCV=ON.");
        return false;
#endif
    }

    if (source_type_ == "image_list") {
#ifdef EDGE_ENABLE_OPENCV
        if (!LoadImageList(path_, image_paths_)) {
            return false;
        }
        width_ = 0;
        height_ = 0;
        channels_ = 3;
        opened_ = true;
        EDGE_LOG_INFO(Name() << " opened image_list path=" << path_
                             << ", images=" << image_paths_.size()
                             << ", max_frames=" << total_frames_);
        return true;
#else
        EDGE_LOG_ERROR(Name() << " requires OpenCV. Reconfigure with -DENABLE_OPENCV=ON.");
        return false;
#endif
    }

    EDGE_LOG_ERROR("Unsupported source_type: " << source_type_);
    return false;
}

bool VideoSource::Read(VideoFrame& frame) {
    if (!opened_) {
        EDGE_LOG_ERROR(Name() << " is not opened");
        return false;
    }

    if (source_type_ == "synthetic") {
        return ReadSynthetic(frame);
    }
    if (source_type_ == "video_file") {
        return ReadVideoFile(frame);
    }
    if (source_type_ == "image_list") {
        return ReadImageList(frame);
    }

    EDGE_LOG_ERROR("Unsupported source_type while reading: " << source_type_);
    return false;
}

bool VideoSource::ReadSynthetic(VideoFrame& frame) {
    if (next_frame_id_ >= total_frames_) {
        return false;
    }

    frame.meta.stream_id = stream_id_;
    frame.meta.frame_id = next_frame_id_;
    frame.meta.width = width_;
    frame.meta.height = height_;
    frame.meta.timestamp_ms = static_cast<int64_t>(next_frame_id_) * 33;
    frame.channels = channels_;

#ifdef EDGE_ENABLE_OPENCV
    frame.image.create(height_, width_, CV_8UC3);
    for (int y = 0; y < height_; ++y) {
        auto* row = frame.image.ptr<uint8_t>(y);
        for (int x = 0; x < width_; ++x) {
            const size_t base = static_cast<size_t>(x) * static_cast<size_t>(channels_);
            row[base + 0] = static_cast<uint8_t>((x + next_frame_id_) % 256);
            row[base + 1] = static_cast<uint8_t>((y + next_frame_id_ * 3) % 256);
            row[base + 2] = static_cast<uint8_t>((x + y + next_frame_id_ * 7) % 256);
        }
    }
#else
    frame.image.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_) * static_cast<size_t>(channels_));
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const size_t base =
                (static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x)) *
                static_cast<size_t>(channels_);
            frame.image[base + 0] = static_cast<uint8_t>((x + next_frame_id_) % 256);
            frame.image[base + 1] = static_cast<uint8_t>((y + next_frame_id_ * 3) % 256);
            frame.image[base + 2] = static_cast<uint8_t>((x + y + next_frame_id_ * 7) % 256);
        }
    }
#endif

    ++next_frame_id_;
    ++frames_read_;

    EDGE_LOG_INFO("VideoSource read stream_id=" << frame.meta.stream_id
                                                << ", frame_id=" << frame.meta.frame_id
                                                << ", timestamp_ms=" << frame.meta.timestamp_ms
                                                << ", read_fps=" << CurrentReadFps());
    return true;
}

bool VideoSource::ReadVideoFile(VideoFrame& frame) {
    if (total_frames_ > 0 && next_frame_id_ >= total_frames_) {
        return false;
    }

#ifdef EDGE_ENABLE_OPENCV
    cv::Mat image;
    if (!capture_.read(image) || image.empty()) {
        return false;
    }

    const double capture_ts_ms = capture_.get(cv::CAP_PROP_POS_MSEC);
    int64_t timestamp_ms = static_cast<int64_t>(capture_ts_ms);
    if (timestamp_ms <= 0 && video_fps_hint_ > 0.0) {
        timestamp_ms = static_cast<int64_t>((static_cast<double>(next_frame_id_) / video_fps_hint_) * 1000.0);
    }

    frame.image = image;
    frame.meta.stream_id = stream_id_;
    frame.meta.frame_id = next_frame_id_;
    frame.meta.width = image.cols;
    frame.meta.height = image.rows;
    frame.meta.timestamp_ms = timestamp_ms;
    frame.channels = image.channels();

    ++next_frame_id_;
    ++frames_read_;

    EDGE_LOG_INFO("VideoSource read stream_id=" << frame.meta.stream_id
                                                << ", frame_id=" << frame.meta.frame_id
                                                << ", timestamp_ms=" << frame.meta.timestamp_ms
                                                << ", read_fps=" << CurrentReadFps());
    return true;
#else
    EDGE_LOG_ERROR(Name() << " cannot read video_file because OpenCV is disabled");
    return false;
#endif
}

bool VideoSource::ReadImageList(VideoFrame& frame) {
#ifdef EDGE_ENABLE_OPENCV
    if (total_frames_ > 0 && next_frame_id_ >= total_frames_) {
        return false;
    }
    if (next_frame_id_ >= static_cast<int>(image_paths_.size())) {
        return false;
    }

    const std::string& image_path = image_paths_[static_cast<std::size_t>(next_frame_id_)];
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        EDGE_LOG_ERROR(Name() << " failed to decode image: " << image_path);
        return false;
    }

    frame.image = image;
    frame.meta.stream_id = stream_id_;
    frame.meta.frame_id = next_frame_id_;
    frame.meta.width = image.cols;
    frame.meta.height = image.rows;
    frame.meta.timestamp_ms = static_cast<int64_t>(next_frame_id_) * 33;
    frame.channels = image.channels();

    ++next_frame_id_;
    ++frames_read_;

    EDGE_LOG_INFO("ImageListSource read stream_id=" << frame.meta.stream_id
                                                    << ", frame_id=" << frame.meta.frame_id
                                                    << ", timestamp_ms=" << frame.meta.timestamp_ms
                                                    << ", path=" << image_path
                                                    << ", read_fps=" << CurrentReadFps());
    return true;
#else
    EDGE_LOG_ERROR(Name() << " cannot read image_list because OpenCV is disabled");
    return false;
#endif
}

double VideoSource::CurrentReadFps() const {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(now - read_start_time_).count();
    if (elapsed_us <= 0) {
        return 0.0;
    }
    return static_cast<double>(frames_read_) * 1000000.0 / static_cast<double>(elapsed_us);
}

int VideoSource::StreamId() const {
    return stream_id_;
}

std::string VideoSource::Name() const {
    return "VideoSource(type=" + source_type_ + ", stream_id=" + std::to_string(stream_id_) + ")";
}

}  // namespace edge
