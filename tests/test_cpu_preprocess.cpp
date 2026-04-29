#include <cassert>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

#include "preprocess/preprocessor.h"

namespace {

bool Close(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1e-4F;
}

edge::VideoFrame MakeFrame(int stream_id, int frame_id, int width, int height, const cv::Scalar& bgr) {
    edge::VideoFrame frame;
    frame.image = cv::Mat(height, width, CV_8UC3, bgr);
    frame.channels = 3;
    frame.meta.stream_id = stream_id;
    frame.meta.frame_id = frame_id;
    frame.meta.width = width;
    frame.meta.height = height;
    frame.meta.timestamp_ms = 0;
    return frame;
}

}  // namespace

int main() {
    edge::CPUPreprocessor preprocessor(640, 640);

    const edge::VideoFrame wide = MakeFrame(0, 0, 1280, 720, cv::Scalar(10, 20, 30));
    const edge::VideoFrame square = MakeFrame(1, 0, 640, 640, cv::Scalar(10, 20, 30));

    edge::TensorBuffer output;
    std::vector<edge::PreprocessMeta> metas;
    assert(preprocessor.RunBatch({wide, square}, output, metas));

    assert(output.shape.size() == 4);
    assert(output.shape[0] == 2);
    assert(output.shape[1] == 3);
    assert(output.shape[2] == 640);
    assert(output.shape[3] == 640);
    assert(output.host_data.size() == 2 * 3 * 640 * 640);
    assert(metas.size() == 2);

    assert(metas[0].original_width == 1280);
    assert(metas[0].original_height == 720);
    assert(metas[0].input_width == 640);
    assert(metas[0].input_height == 640);
    assert(Close(metas[0].scale, 0.5F));
    assert(metas[0].pad_x == 0);
    assert(metas[0].pad_y == 140);

    assert(Close(metas[1].scale, 1.0F));
    assert(metas[1].pad_x == 0);
    assert(metas[1].pad_y == 0);

    const size_t sample_y = 320;
    const size_t sample_x = 320;
    const size_t h = 640;
    const size_t w = 640;
    const size_t sample_base = sample_y * w + sample_x;
    const float r = output.host_data[0 * h * w + sample_base];
    const float g = output.host_data[1 * h * w + sample_base];
    const float b = output.host_data[2 * h * w + sample_base];
    assert(Close(r, 30.0F / 255.0F));
    assert(Close(g, 20.0F / 255.0F));
    assert(Close(b, 10.0F / 255.0F));

    return 0;
}
