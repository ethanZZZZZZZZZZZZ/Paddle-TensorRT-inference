#include "pipeline/batch_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <string>
#include <utility>

#include "common/logging.h"

namespace edge {

BatchScheduler::BatchScheduler(int max_batch_size, bool enable_dynamic_batch, int timeout_ms)
    : max_batch_size_(std::max(1, max_batch_size)),
      enable_dynamic_batch_(enable_dynamic_batch),
      timeout_ms_(std::max(0, timeout_ms)) {
    pending_.reserve(static_cast<size_t>(max_batch_size_));
}

void BatchScheduler::Enqueue(VideoFrame frame, double video_decode_ms) {
    if (pending_.empty()) {
        pending_start_time_ = std::chrono::steady_clock::now();
        pending_created_timestamp_us_ = NowUs();
    }

    pending_.push_back(PendingFrame{std::move(frame), video_decode_ms});
}

bool BatchScheduler::PopReadyBatch(FrameBatch& batch) {
    if (!ShouldFlush()) {
        return false;
    }

    std::string trigger_reason;
    if (!enable_dynamic_batch_) {
        trigger_reason = "single_frame";
    } else if (PendingSize() >= max_batch_size_) {
        trigger_reason = "max_batch_size";
    } else {
        trigger_reason = "timeout";
    }

    batch = MakeBatch(trigger_reason);
    return true;
}

bool BatchScheduler::Flush(FrameBatch& batch) {
    if (pending_.empty()) {
        return false;
    }

    batch = MakeBatch("drain");
    return true;
}

bool BatchScheduler::HasPending() const {
    return !pending_.empty();
}

int BatchScheduler::PendingSize() const {
    return static_cast<int>(pending_.size());
}

int BatchScheduler::MaxBatchSize() const {
    return max_batch_size_;
}

bool BatchScheduler::DynamicBatchEnabled() const {
    return enable_dynamic_batch_;
}

int BatchScheduler::TimeoutMs() const {
    return timeout_ms_;
}

std::string BatchScheduler::Name() const {
    return "BatchScheduler(max_batch_size=" + std::to_string(max_batch_size_) +
           ", dynamic=" + (enable_dynamic_batch_ ? "true" : "false") +
           ", timeout_ms=" + std::to_string(timeout_ms_) + ")";
}

bool BatchScheduler::ShouldFlush() const {
    if (pending_.empty()) {
        return false;
    }
    if (!enable_dynamic_batch_) {
        return true;
    }
    if (PendingSize() >= max_batch_size_) {
        return true;
    }
    return PendingWaitMs() >= static_cast<double>(timeout_ms_);
}

FrameBatch BatchScheduler::MakeBatch(const std::string& trigger_reason) {
    FrameBatch batch;
    batch.batch_id = next_batch_id_++;
    batch.created_timestamp_us = pending_created_timestamp_us_;
    batch.batch_wait_ms = PendingWaitMs();
    batch.trigger_reason = trigger_reason;
    batch.frames.reserve(pending_.size());

    for (auto& item : pending_) {
        batch.video_decode_ms += item.video_decode_ms;
        batch.frames.push_back(std::move(item.frame));
    }

    EDGE_LOG_INFO("[BatchScheduler] emit batch_id=" << batch.batch_id
                                                    << ", actual_batch_size="
                                                    << batch.ActualBatchSize()
                                                    << ", max_batch_size="
                                                    << max_batch_size_
                                                    << ", trigger="
                                                    << batch.trigger_reason
                                                    << ", batch_wait_ms="
                                                    << batch.batch_wait_ms
                                                    << ", video_decode_ms="
                                                    << batch.video_decode_ms);

    pending_.clear();
    pending_created_timestamp_us_ = 0;
    return batch;
}

double BatchScheduler::PendingWaitMs() const {
    if (pending_.empty()) {
        return 0.0;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now - pending_start_time_).count();
    return static_cast<double>(us) / 1000.0;
}

int64_t BatchScheduler::NowUs() const {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

}  // namespace edge
