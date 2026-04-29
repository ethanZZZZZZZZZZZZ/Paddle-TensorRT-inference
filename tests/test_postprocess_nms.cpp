#include <cassert>
#include <vector>

#include "postprocess/postprocessor.h"

int main() {
    std::vector<edge::Detection> detections;

    edge::Detection high;
    high.stream_id = 0;
    high.frame_id = 0;
    high.class_id = 1;
    high.score = 0.95F;
    high.x1 = 10.0F;
    high.y1 = 10.0F;
    high.x2 = 110.0F;
    high.y2 = 110.0F;
    detections.push_back(high);

    edge::Detection overlap = high;
    overlap.score = 0.80F;
    overlap.x1 = 20.0F;
    overlap.y1 = 20.0F;
    overlap.x2 = 120.0F;
    overlap.y2 = 120.0F;
    detections.push_back(overlap);

    edge::Detection other_class = overlap;
    other_class.class_id = 2;
    other_class.score = 0.70F;
    detections.push_back(other_class);

    const auto kept = edge::CPUPostprocessor::Nms(detections, 0.45F, 100);
    assert(kept.size() == 2);
    assert(kept[0].score == 0.95F);
    assert(kept[1].class_id == 2);

    return 0;
}
