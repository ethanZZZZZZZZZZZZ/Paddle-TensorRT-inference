#include <cassert>
#include <cmath>

#include "postprocess/postprocessor.h"

namespace {

bool Close(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1e-4F;
}

}  // namespace

int main() {
    edge::FrameMeta frame_meta;
    frame_meta.stream_id = 2;
    frame_meta.frame_id = 7;
    frame_meta.width = 1280;
    frame_meta.height = 720;

    edge::PreprocessMeta preprocess_meta;
    preprocess_meta.original_width = 1280;
    preprocess_meta.original_height = 720;
    preprocess_meta.input_width = 640;
    preprocess_meta.input_height = 640;
    preprocess_meta.scale = 0.5F;
    preprocess_meta.pad_x = 0;
    preprocess_meta.pad_y = 140;

    edge::Detection input_space;
    input_space.class_id = 1;
    input_space.score = 0.9F;
    input_space.x1 = 100.0F;
    input_space.y1 = 190.0F;
    input_space.x2 = 300.0F;
    input_space.y2 = 390.0F;

    const edge::Detection mapped =
        edge::CPUPostprocessor::MapBoxToOriginal(input_space, frame_meta, preprocess_meta);

    assert(mapped.stream_id == 2);
    assert(mapped.frame_id == 7);
    assert(mapped.class_id == 1);
    assert(Close(mapped.x1, 200.0F));
    assert(Close(mapped.y1, 100.0F));
    assert(Close(mapped.x2, 600.0F));
    assert(Close(mapped.y2, 500.0F));

    return 0;
}
