#include "postprocess/gpu_decode_pre_nms.h"

#ifdef EDGE_ENABLE_CUDA

#include <algorithm>
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
    output.shape = {2, 5, 6};
    output.host_data = {
        100.0F, 190.0F, 300.0F, 390.0F, 0.90F, 1.0F,
        120.0F, 210.0F, 280.0F, 360.0F, 0.20F, 1.0F,
        140.0F, 220.0F, 320.0F, 420.0F, 0.80F, 2.0F,
        160.0F, 240.0F, 340.0F, 440.0F, 0.70F, 2.0F,
        180.0F, 260.0F, 360.0F, 460.0F, 0.10F, 0.0F,
        90.0F,  180.0F, 250.0F, 350.0F, 0.95F, 0.0F,
        110.0F, 200.0F, 260.0F, 370.0F, 0.40F, 1.0F,
        130.0F, 220.0F, 290.0F, 390.0F, 0.30F, 1.0F,
        150.0F, 240.0F, 310.0F, 410.0F, 0.29F, 2.0F,
        170.0F, 260.0F, 330.0F, 430.0F, 0.85F, 2.0F,
    };
    return output;
}

std::vector<edge::FrameMeta> MakeFrameMetas() {
    std::vector<edge::FrameMeta> metas(2);
    metas[0].stream_id = 0;
    metas[0].frame_id = 10;
    metas[0].width = 1280;
    metas[0].height = 720;
    metas[1].stream_id = 1;
    metas[1].frame_id = 20;
    metas[1].width = 1280;
    metas[1].height = 720;
    return metas;
}

std::vector<edge::PreprocessMeta> MakePreprocessMetas() {
    std::vector<edge::PreprocessMeta> metas(2);
    for (auto& meta : metas) {
        meta.original_width = 1280;
        meta.original_height = 720;
        meta.input_width = 640;
        meta.input_height = 640;
        meta.scale = 0.5F;
        meta.pad_x = 0;
        meta.pad_y = 140;
    }
    return metas;
}

void SortDetections(std::vector<edge::Detection>& detections) {
    std::sort(detections.begin(), detections.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.stream_id != rhs.stream_id) {
            return lhs.stream_id < rhs.stream_id;
        }
        if (lhs.frame_id != rhs.frame_id) {
            return lhs.frame_id < rhs.frame_id;
        }
        if (lhs.class_id != rhs.class_id) {
            return lhs.class_id < rhs.class_id;
        }
        if (!Close(lhs.score, rhs.score)) {
            return lhs.score > rhs.score;
        }
        if (!Close(lhs.x1, rhs.x1)) {
            return lhs.x1 < rhs.x1;
        }
        return lhs.y1 < rhs.y1;
    });
}

void CompareDetections(std::vector<edge::Detection> cpu, std::vector<edge::Detection> gpu) {
    SortDetections(cpu);
    SortDetections(gpu);
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
    config.score_threshold = 0.30F;
    config.nms_threshold = 0.45F;
    config.top_k = 10;

    const edge::TensorBuffer model_output = MakeMockOutput();
    const auto frame_metas = MakeFrameMetas();
    const auto preprocess_metas = MakePreprocessMetas();

    edge::CPUPostprocessor cpu_postprocessor(config);
    std::vector<edge::Detection> cpu_decoded;
    assert(cpu_postprocessor.DecodePreNms(model_output, frame_metas, preprocess_metas, cpu_decoded));

    cudaStream_t stream = nullptr;
    assert(cudaStreamCreate(&stream) == cudaSuccess);

    edge::GpuDecodePreNMS gpu_decode(config);
    std::vector<edge::Detection> gpu_decoded;
    double gpu_ms = 0.0;
    assert(gpu_decode.Run(model_output,
                          frame_metas,
                          preprocess_metas,
                          640,
                          640,
                          stream,
                          gpu_decoded,
                          &gpu_ms));
    assert(cudaStreamDestroy(stream) == cudaSuccess);

    CompareDetections(cpu_decoded, gpu_decoded);

    config.top_k = 2;
    edge::GpuDecodePreNMS gpu_decode_topk(config);
    assert(cudaStreamCreate(&stream) == cudaSuccess);
    gpu_decoded.clear();
    assert(gpu_decode_topk.Run(model_output,
                               frame_metas,
                               preprocess_metas,
                               640,
                               640,
                               stream,
                               gpu_decoded,
                               &gpu_ms));
    assert(cudaStreamDestroy(stream) == cudaSuccess);
    assert(gpu_decoded.size() <= 4);
    return 0;
}

#endif  // EDGE_ENABLE_CUDA
