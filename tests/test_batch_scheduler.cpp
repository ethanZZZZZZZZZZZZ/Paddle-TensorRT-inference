#include "pipeline/batch_scheduler.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

edge::VideoFrame MakeFrame(int stream_id, int frame_id) {
    edge::VideoFrame frame;
    frame.meta.stream_id = stream_id;
    frame.meta.frame_id = frame_id;
    frame.meta.width = 16;
    frame.meta.height = 16;
    frame.meta.timestamp_ms = static_cast<int64_t>(frame_id) * 33;
    frame.channels = 3;

#ifdef EDGE_ENABLE_OPENCV
    frame.image.create(frame.meta.height, frame.meta.width, CV_8UC3);
#else
    frame.image.assign(
        static_cast<size_t>(frame.meta.width) * static_cast<size_t>(frame.meta.height) *
            static_cast<size_t>(frame.channels),
        0);
#endif
    return frame;
}

void TestMaxBatchTrigger() {
    edge::BatchScheduler scheduler(3, true, 100);
    edge::FrameBatch batch;

    scheduler.Enqueue(MakeFrame(0, 0), 0.1);
    assert(!scheduler.PopReadyBatch(batch));
    scheduler.Enqueue(MakeFrame(1, 0), 0.2);
    assert(!scheduler.PopReadyBatch(batch));
    scheduler.Enqueue(MakeFrame(2, 0), 0.3);
    assert(scheduler.PopReadyBatch(batch));

    assert(batch.batch_id == 0);
    assert(batch.ActualBatchSize() == 3);
    assert(batch.trigger_reason == "max_batch_size");
    assert(batch.frames[0].meta.stream_id == 0);
    assert(batch.frames[1].meta.stream_id == 1);
    assert(batch.frames[2].meta.stream_id == 2);
    assert(batch.video_decode_ms > 0.0);
}

void TestImmediateTimeoutTrigger() {
    edge::BatchScheduler scheduler(4, true, 0);
    edge::FrameBatch batch;

    scheduler.Enqueue(MakeFrame(5, 7), 0.4);
    assert(scheduler.PopReadyBatch(batch));
    assert(batch.ActualBatchSize() == 1);
    assert(batch.trigger_reason == "timeout");
    assert(batch.frames[0].meta.stream_id == 5);
    assert(batch.frames[0].meta.frame_id == 7);
}

void TestSingleFrameMode() {
    edge::BatchScheduler scheduler(8, false, 100);
    edge::FrameBatch batch;

    scheduler.Enqueue(MakeFrame(3, 4), 0.5);
    assert(scheduler.PopReadyBatch(batch));
    assert(batch.ActualBatchSize() == 1);
    assert(batch.trigger_reason == "single_frame");
}

}  // namespace

int main() {
    TestMaxBatchTrigger();
    TestImmediateTimeoutTrigger();
    TestSingleFrameMode();
    return 0;
}
