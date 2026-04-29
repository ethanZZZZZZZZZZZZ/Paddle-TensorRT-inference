#include "postprocess/gpu_decode_pre_nms.h"

#ifdef EDGE_ENABLE_CUDA

#include <cassert>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "postprocess/postprocessor.h"

namespace {

bool Close(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1e-4F;
}

edge::TensorBuffer MakeMockOutput() {
    edge::TensorBuffer output;
    output.shape = {1, 5, 6};
    output.host_data = {
        100.0F, 190.0F, 300.0F, 390.0F, 0.95F, 1.0F,
        110.0F, 200.0F, 310.0F, 400.0F, 0.80F, 1.0F,
        320.0F, 260.0F, 420.0F, 360.0F, 0.70F, 1.0F,
        115.0F, 205.0F, 315.0F, 405.0F, 0.60F, 2.0F,
        10.0F,  20.0F,  30.0F,  40.0F,  0.10F, 0.0F,
    };
    return output;
}

std::vector<edge::FrameMeta> MakeFrameMetas() {
    std::vector<edge::FrameMeta> metas(1);
    metas[0].stream_id = 2;
    metas[0].frame_id = 42;
    metas[0].width = 1280;
    metas[0].height = 720;
    return metas;
}

std::vector<edge::PreprocessMeta> MakePreprocessMetas() {
    std::vector<edge::PreprocessMeta> metas(1);
    metas[0].original_width = 1280;
    metas[0].original_height = 720;
    metas[0].input_width = 640;
    metas[0].input_height = 640;
    metas[0].scale = 0.5F;
    metas[0].pad_x = 0;
    metas[0].pad_y = 140;
    return metas;
}

void CompareDetections(const std::vector<edge::Detection>& cpu, const std::vector<edge::Detection>& gpu) {
    assert(cpu.size() == gpu.size());
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        assert(cpu[i].stream_id == gpu[i].stream_id);
        assert(cpu[i].frame_id == gpu[i].frame_id);
        assert(cpu[i].class_id == gpu[i].class_id);
        assert(Close(cpu[i].score, gpu[i].score));
        assert(Close(cpu[i].x1, gpu[i].x1));
        assert(Close(cpu[i].y1, gpu[i].y1));
        assert(Close(cpu[i].x2, gpu[i].x2));
        assert(Close(cpu[i].y2, gpu[i].y2));
    }
}

}  // namespace

int main() {
    edge::PostprocessConfig config;
    config.mode = "mock_yolo";
    config.decode_backend = "gpu";
    config.nms_backend = "gpu";
    config.score_threshold = 0.30F;
    config.nms_threshold = 0.45F;
    config.top_k = 10;

    const edge::TensorBuffer model_output = MakeMockOutput();
    const auto frame_metas = MakeFrameMetas();
    const auto preprocess_metas = MakePreprocessMetas();

    edge::CPUPostprocessor cpu_postprocessor(config);
    std::vector<edge::Detection> cpu_detections;
    edge::PostprocessTiming cpu_timing;
    assert(cpu_postprocessor.Run(model_output, frame_metas, preprocess_metas, cpu_detections, &cpu_timing));

    cudaStream_t stream = nullptr;
    assert(cudaStreamCreate(&stream) == cudaSuccess);

    edge::GpuDecodePreNMS gpu_postprocessor(config);
    std::vector<edge::Detection> gpu_detections;
    double decode_ms = 0.0;
    double nms_ms = 0.0;
    assert(gpu_postprocessor.RunWithGpuNms(model_output,
                                           frame_metas,
                                           preprocess_metas,
                                           640,
                                           640,
                                           stream,
                                           gpu_detections,
                                           &decode_ms,
                                           &nms_ms));
    assert(cudaStreamDestroy(stream) == cudaSuccess);

    CompareDetections(cpu_detections, gpu_detections);
    return 0;
}

#endif  // EDGE_ENABLE_CUDA
