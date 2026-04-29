#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "common/types.h"

namespace edge {

struct FrameBatch {
    std::vector<VideoFrame> frames;
    int batch_id = 0;
    int64_t created_timestamp_us = 0;
    double batch_wait_ms = 0.0;
    double video_decode_ms = 0.0;
    std::string trigger_reason;

    int ActualBatchSize() const {
        return static_cast<int>(frames.size());
    }
};

class BatchScheduler {
public:
    BatchScheduler(int max_batch_size, bool enable_dynamic_batch, int timeout_ms);

    void Enqueue(VideoFrame frame, double video_decode_ms);
    bool PopReadyBatch(FrameBatch& batch);
    bool Flush(FrameBatch& batch);
    bool HasPending() const;
    int PendingSize() const;
    int MaxBatchSize() const;
    bool DynamicBatchEnabled() const;
    int TimeoutMs() const;
    std::string Name() const;

private:
    struct PendingFrame {
        VideoFrame frame;
        double video_decode_ms = 0.0;
    };

    bool ShouldFlush() const;
    FrameBatch MakeBatch(const std::string& trigger_reason);
    double PendingWaitMs() const;
    int64_t NowUs() const;

    int max_batch_size_ = 1;
    bool enable_dynamic_batch_ = false;
    int timeout_ms_ = 0;
    int next_batch_id_ = 0;
    int64_t pending_created_timestamp_us_ = 0;
    std::chrono::steady_clock::time_point pending_start_time_;
    std::vector<PendingFrame> pending_;
};

}  // namespace edge
