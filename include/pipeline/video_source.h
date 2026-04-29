#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "common/types.h"

#ifdef EDGE_ENABLE_OPENCV
#include <opencv2/videoio.hpp>
#endif

namespace edge {

class VideoSource {
public:
    VideoSource(
        int stream_id,
        std::string source_type,
        std::string path,
        int width,
        int height,
        int channels,
        int total_frames);

    bool Open();
    bool Read(VideoFrame& frame);
    std::string Name() const;
    int StreamId() const;

private:
    bool ReadSynthetic(VideoFrame& frame);
    bool ReadVideoFile(VideoFrame& frame);
    bool ReadImageList(VideoFrame& frame);
    double CurrentReadFps() const;

    int stream_id_ = 0;
    std::string source_type_;
    std::string path_;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 3;
    int total_frames_ = 0;
    int next_frame_id_ = 0;
    int frames_read_ = 0;
    double video_fps_hint_ = 0.0;
    std::chrono::steady_clock::time_point read_start_time_;
    bool opened_ = false;

#ifdef EDGE_ENABLE_OPENCV
    cv::VideoCapture capture_;
    std::vector<std::string> image_paths_;
#endif
};

}  // namespace edge
