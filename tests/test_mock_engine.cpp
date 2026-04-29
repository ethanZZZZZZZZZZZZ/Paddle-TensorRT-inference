#include <cassert>
#include <vector>

#include "common/config.h"
#include "common/types.h"
#include "inference/mock_infer_engine.h"

int main() {
    edge::AppConfig config;
    config.model.input_width = 640;
    config.model.input_height = 640;
    config.model.num_classes = 3;
    config.model.mock_num_boxes = 2;

    edge::MockInferEngine engine;
    assert(engine.Init(config));

    edge::TensorBuffer input;
    input.shape = {2, 3, 640, 640};
    input.host_data.assign(2 * 3 * 640 * 640, 0.5F);

    edge::TensorBuffer output;
    assert(engine.Infer(input, output));
    assert(output.shape.size() == 3);
    assert(output.shape[0] == 2);
    assert(output.shape[1] == 2);
    assert(output.shape[2] == 6);
    assert(output.host_data.size() == 2 * 2 * 6);
    assert(output.host_data[4] > output.host_data[10]);

    return 0;
}
