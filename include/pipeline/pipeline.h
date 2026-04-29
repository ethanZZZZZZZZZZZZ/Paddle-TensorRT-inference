#pragma once

#include <chrono>
#include <fstream>
#include <memory>
#include <vector>

#include "common/config.h"
#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)
#include "cuda/cuda_runtime_utils.h"
#endif
#include "inference/infer_engine.h"
#include "inference/predictor_pool.h"
#include "pipeline/batch_scheduler.h"
#include "pipeline/video_source.h"
#include "postprocess/postprocessor.h"
#ifdef EDGE_ENABLE_CUDA
#include "postprocess/gpu_decode_pre_nms.h"
#include "preprocess/gpu_preprocessor.h"
#endif
#ifdef EDGE_ENABLE_TENSORRT_PLUGIN
#include "postprocess/trt_postprocess_engine.h"
#endif
#include "preprocess/preprocessor.h"
#include "profiling/profiler.h"

namespace edge {

class Pipeline {
public:
    explicit Pipeline(const Config& config);

    bool Init();
    bool Run();

private:
    struct PreparedBatch {
        FrameBatch batch;
        TensorBuffer input_tensor;
        std::vector<PreprocessMeta> preprocess_metas;
        std::vector<FrameMeta> frame_metas;
        std::chrono::steady_clock::time_point process_start_time;
        double preprocess_ms = 0.0;
        double cpu_preprocess_ms = 0.0;
        double gpu_preprocess_ms = 0.0;
        double d2h_copy_ms = 0.0;
    };

    bool ProcessBatch(
        const FrameBatch& batch,
        std::ofstream* result_file,
        IterationMetrics& metrics);
    bool RunWithPredictorPool();
#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)
    bool AcquireCudaStream(const std::string& stage_name, CudaStreamLease& lease);
#endif
    bool PreprocessBatch(const FrameBatch& batch, PreparedBatch& prepared);
    bool PostprocessPreparedBatch(
        const PreparedBatch& prepared,
        const TensorBuffer* host_output,
        const InferOutput* infer_output,
        double inference_ms,
        double inference_queue_wait_ms,
        int predictor_worker_id,
        std::ofstream* result_file,
        IterationMetrics& metrics);

    AppConfig config_;
    std::vector<std::unique_ptr<VideoSource>> sources_;
    BatchScheduler batch_scheduler_;
    CPUPreprocessor preprocessor_;
#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)
    std::unique_ptr<CudaStreamPool> cuda_stream_pool_;
#endif
#ifdef EDGE_ENABLE_CUDA
    std::unique_ptr<GpuPreprocessor> gpu_preprocessor_;
    std::unique_ptr<GpuDecodePreNMS> gpu_decode_pre_nms_;
#endif
#ifdef EDGE_ENABLE_TENSORRT_PLUGIN
    std::unique_ptr<TrtPostprocessEngine> trt_postprocess_engine_;
#endif
    CPUPostprocessor postprocessor_;
    std::unique_ptr<InferEngine> infer_engine_;
    std::unique_ptr<PredictorPool> predictor_pool_;
    Profiler profiler_;
};

}  // namespace edge
