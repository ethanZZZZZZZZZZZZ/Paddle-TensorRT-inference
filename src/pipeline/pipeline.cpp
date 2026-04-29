#include "pipeline/pipeline.h"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef EDGE_ENABLE_TENSORRT_PLUGIN
#include <cuda_runtime_api.h>
#endif

#include "common/logging.h"
#include "inference/mock_infer_engine.h"
#ifdef EDGE_ENABLE_PADDLE
#include "inference/paddle_infer_engine.h"
#endif
#include "profiling/nvtx_utils.h"

namespace edge {
namespace {

double ElapsedMs(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return static_cast<double>(us) / 1000.0;
}

}  // namespace

Pipeline::Pipeline(const Config& config)
    : config_(config.Data()),
      batch_scheduler_(config_.infer.batch_size,
                       config_.infer.enable_dynamic_batch,
                       config_.infer.dynamic_batch_timeout_ms),
      preprocessor_(config_.model.input_width, config_.model.input_height),
      postprocessor_(config_.postprocess) {}

bool Pipeline::Init() {
    EDGE_LOG_INFO("Pipeline init started");
    EDGE_LOG_INFO("Requested infer.backend=" << config_.infer.backend);

    if (config_.infer.predictor_pool_size > 1) {
        predictor_pool_ = std::make_unique<PredictorPool>();
        EDGE_LOG_INFO("Predictor pool requested, size=" << config_.infer.predictor_pool_size);
    }

    if (!predictor_pool_) {
        if (config_.infer.backend == "mock") {
            infer_engine_ = std::make_unique<MockInferEngine>();
            EDGE_LOG_INFO("Paddle Inference enabled: false");
            EDGE_LOG_INFO("TensorRT enabled: false");
#ifdef EDGE_ENABLE_PADDLE
        } else if (config_.infer.backend == "paddle" || config_.infer.backend == "paddle_trt") {
            infer_engine_ = std::make_unique<PaddleInferEngine>();
#endif
        } else {
            EDGE_LOG_ERROR("Unsupported infer.backend: " << config_.infer.backend);
            return false;
        }
    }

    sources_.clear();
    sources_.reserve(static_cast<size_t>(config_.input.num_streams));
    for (int stream_id = 0; stream_id < config_.input.num_streams; ++stream_id) {
        sources_.push_back(std::make_unique<VideoSource>(stream_id,
                                                         config_.input.source_type,
                                                         config_.input.path,
                                                         config_.input.synthetic_width,
                                                         config_.input.synthetic_height,
                                                         config_.input.synthetic_channels,
                                                         config_.input.num_frames));
    }

    for (auto& source : sources_) {
        if (!source->Open()) {
            EDGE_LOG_ERROR("Pipeline init failed while opening source: " << source->Name());
            return false;
        }
    }
    if (predictor_pool_) {
        if (!predictor_pool_->Init(config_)) {
            EDGE_LOG_ERROR("Pipeline init failed while initializing predictor pool");
            return false;
        }
    } else {
        if (!infer_engine_->Init(config_)) {
            EDGE_LOG_ERROR("Pipeline init failed while initializing infer engine: " << infer_engine_->Name());
            return false;
        }
    }
#ifdef EDGE_ENABLE_CUDA
    if (config_.preprocess.type == "gpu") {
        gpu_preprocessor_ =
            std::make_unique<GpuPreprocessor>(config_.model.input_width, config_.model.input_height);
    }
    if (config_.postprocess.decode_backend == "gpu") {
        gpu_decode_pre_nms_ =
            std::make_unique<GpuDecodePreNMS>(config_.postprocess, config_.cuda.enable_pinned_memory);
    }
#endif
#ifdef EDGE_ENABLE_TENSORRT_PLUGIN
    if (config_.postprocess.decode_backend == "trt_plugin") {
        trt_postprocess_engine_ = std::make_unique<TrtPostprocessEngine>(
            config_.postprocess,
            config_.model.input_width,
            config_.model.input_height,
            config_.cuda.enable_pinned_memory);
    }
#endif
#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)
    const bool needs_cuda_stream_pool =
#ifdef EDGE_ENABLE_CUDA
        config_.preprocess.type == "gpu" || config_.postprocess.decode_backend == "gpu" ||
#else
        false ||
#endif
#ifdef EDGE_ENABLE_TENSORRT_PLUGIN
        config_.postprocess.decode_backend == "trt_plugin";
#else
        false;
#endif
    if (needs_cuda_stream_pool) {
        cuda_stream_pool_ = std::make_unique<CudaStreamPool>();
        if (!cuda_stream_pool_->Init(config_.cuda.stream_pool_size)) {
            EDGE_LOG_ERROR("Pipeline init failed while initializing CUDA stream pool");
            return false;
        }
    }
#endif

    EDGE_LOG_INFO("Pipeline components:");
    EDGE_LOG_INFO("  source_count=" << sources_.size());
    for (const auto& source : sources_) {
        EDGE_LOG_INFO("  source=" << source->Name());
    }
    EDGE_LOG_INFO("  preprocess.type=" << config_.preprocess.type);
    EDGE_LOG_INFO("  cpu_preprocessor=" << preprocessor_.Name());
#ifdef EDGE_ENABLE_CUDA
    if (gpu_preprocessor_) {
        EDGE_LOG_INFO("  gpu_preprocessor=" << gpu_preprocessor_->Name());
    }
#endif
    EDGE_LOG_INFO("  batch_scheduler=" << batch_scheduler_.Name());
#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)
    if (cuda_stream_pool_) {
        EDGE_LOG_INFO("  cuda_stream_pool=" << cuda_stream_pool_->Name());
    }
#endif
    if (predictor_pool_) {
        EDGE_LOG_INFO("  predictor_pool=" << predictor_pool_->Name());
    } else {
        EDGE_LOG_INFO("  infer_engine=" << infer_engine_->Name());
    }
    EDGE_LOG_INFO("  infer.backend=" << config_.infer.backend);
    EDGE_LOG_INFO("  postprocessor=" << postprocessor_.Name());
#ifdef EDGE_ENABLE_CUDA
    if (gpu_decode_pre_nms_) {
        EDGE_LOG_INFO("  gpu_decode_pre_nms=" << gpu_decode_pre_nms_->Name());
    }
#endif
#ifdef EDGE_ENABLE_TENSORRT_PLUGIN
    if (trt_postprocess_engine_) {
        EDGE_LOG_INFO("  trt_postprocess_engine=" << trt_postprocess_engine_->Name());
    }
#endif
    EDGE_LOG_INFO("  postprocess.mode=" << config_.postprocess.mode);
    EDGE_LOG_INFO("  postprocess.decode_backend=" << config_.postprocess.decode_backend);
    EDGE_LOG_INFO("  postprocess.nms_backend=" << config_.postprocess.nms_backend);
    EDGE_LOG_INFO("  precision=" << config_.infer.precision);
    EDGE_LOG_INFO("  batch_size=" << config_.infer.batch_size);
    EDGE_LOG_INFO("  enable_dynamic_batch="
                  << (config_.infer.enable_dynamic_batch ? "true" : "false"));
    EDGE_LOG_INFO("  dynamic_batch_timeout_ms="
                  << config_.infer.dynamic_batch_timeout_ms);
    EDGE_LOG_INFO("  predictor_pool_size=" << config_.infer.predictor_pool_size);
    EDGE_LOG_INFO("  cuda.stream_pool_size=" << config_.cuda.stream_pool_size);
    EDGE_LOG_INFO("  cuda.enable_pinned_memory="
                  << (config_.cuda.enable_pinned_memory ? "true" : "false"));
    EDGE_LOG_INFO("  profile.enable_timer=" << (config_.profile.enable_timer ? "true" : "false"));
    EDGE_LOG_INFO("  output.save_result=" << (config_.output.save_result ? "true" : "false"));
    EDGE_LOG_INFO("  benchmark.warmup_iters=" << config_.benchmark.warmup_iters);
    EDGE_LOG_INFO("  benchmark.benchmark_iters=" << config_.benchmark.benchmark_iters);
    EDGE_LOG_INFO("  benchmark.output_csv=" << config_.benchmark.output_csv);
    EDGE_LOG_INFO("  trt_analysis.enable=" << (config_.trt_analysis.enable ? "true" : "false"));
    if (config_.trt_analysis.enable) {
        EDGE_LOG_INFO("  trt_analysis.log_path=" << config_.trt_analysis.log_path);
        EDGE_LOG_INFO("  trt_analysis.report_json=" << config_.trt_analysis.report_json);
        EDGE_LOG_INFO("  trt_analysis.report_md=" << config_.trt_analysis.report_md);
    }
    EDGE_LOG_INFO("Pipeline init finished");
    return true;
}

bool Pipeline::Run() {
    if (predictor_pool_) {
        return RunWithPredictorPool();
    }

    if (!infer_engine_) {
        EDGE_LOG_ERROR("Pipeline must be initialized before Run()");
        return false;
    }

    EDGE_LOG_INFO("Pipeline run started");
    int processed_frames = 0;
    std::ofstream result_file;

    if (config_.output.save_result) {
        result_file.open(config_.output.result_path);
        if (!result_file.is_open()) {
            EDGE_LOG_ERROR("Failed to open output.result_path: " << config_.output.result_path);
            return false;
        }
        EDGE_LOG_INFO("Saving mock results to " << config_.output.result_path);
    }

    std::vector<bool> active(sources_.size(), true);
    int active_count = static_cast<int>(sources_.size());
    size_t next_source_index = 0;
    int seen_iters = 0;
    int benchmark_iters = 0;
    int processed_batches = 0;
    bool reached_benchmark_limit = false;

    auto handle_batch = [&](const FrameBatch& batch) -> bool {
        IterationMetrics metrics;
        if (!ProcessBatch(batch,
                          result_file.is_open() ? &result_file : nullptr,
                          metrics)) {
            return false;
        }

        processed_frames += batch.ActualBatchSize();
        ++processed_batches;

        const bool warmup = seen_iters < config_.benchmark.warmup_iters;
        ++seen_iters;
        if (warmup) {
            EDGE_LOG_INFO("[Benchmark] warmup iter=" << seen_iters
                                                    << ", batch_id=" << batch.batch_id
                                                    << ", actual_batch_size="
                                                    << batch.ActualBatchSize()
                                                    << ", e2e_ms=" << metrics.e2e_ms);
            return true;
        }

        ++benchmark_iters;
        metrics.iter = benchmark_iters;
        profiler_.AddIteration(metrics);
        if (benchmark_iters == 1 || benchmark_iters % 10 == 0) {
            profiler_.ReportToStdout(std::cout);
        }

        if (config_.benchmark.benchmark_iters > 0 &&
            benchmark_iters >= config_.benchmark.benchmark_iters) {
            reached_benchmark_limit = true;
        }
        return true;
    };

    while ((active_count > 0 || batch_scheduler_.HasPending()) && !reached_benchmark_limit) {
        FrameBatch ready_batch;
        if (batch_scheduler_.PopReadyBatch(ready_batch)) {
            if (!handle_batch(ready_batch)) {
                return false;
            }
            continue;
        }

        if (active_count <= 0) {
            if (batch_scheduler_.Flush(ready_batch)) {
                if (!handle_batch(ready_batch)) {
                    return false;
                }
            }
            continue;
        }

        bool read_frame = false;
        for (size_t scan = 0; scan < sources_.size(); ++scan) {
            const size_t index = next_source_index;
            next_source_index = (next_source_index + 1) % sources_.size();
            if (!active[index]) {
                continue;
            }

            VideoFrame frame;
            Timer video_decode_timer;
            video_decode_timer.Tic();
            bool read_ok = false;
            {
                PROFILE_RANGE("video_decode");
                read_ok = sources_[index]->Read(frame);
            }
            const double video_decode_ms = video_decode_timer.TocMs();
            if (!read_ok) {
                active[index] = false;
                --active_count;
                EDGE_LOG_INFO("VideoSource exhausted stream_id=" << sources_[index]->StreamId());
                continue;
            }

            EDGE_LOG_INFO("Stage video_source done stream_id=" << frame.meta.stream_id
                                                               << ", frame_id="
                                                               << frame.meta.frame_id
                                                               << ", timestamp_ms="
                                                               << frame.meta.timestamp_ms);
            batch_scheduler_.Enqueue(std::move(frame), video_decode_ms);
            read_frame = true;
            break;
        }

        if (!read_frame && active_count <= 0) {
            continue;
        }
    }

    EDGE_LOG_INFO("Pipeline run finished, processed_frames=" << processed_frames
                                                             << ", processed_batches="
                                                             << processed_batches);
    if (profiler_.Count() > 0) {
        profiler_.ReportToStdout(std::cout);
        if (!profiler_.SaveCsv(config_.benchmark.output_csv)) {
            return false;
        }
    } else {
        EDGE_LOG_WARN("No benchmark samples were collected; CSV will not be written");
    }
    return processed_frames > 0;
}

#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)
bool Pipeline::AcquireCudaStream(const std::string& stage_name, CudaStreamLease& lease) {
    if (!cuda_stream_pool_) {
        EDGE_LOG_ERROR("CUDA stream pool was not initialized for stage=" << stage_name);
        return false;
    }
    lease = cuda_stream_pool_->Acquire();
    if (!lease.Valid()) {
        EDGE_LOG_ERROR("Failed to acquire CUDA stream for stage=" << stage_name);
        return false;
    }
    EDGE_LOG_INFO("[CudaStream] stage=" << stage_name
                                        << ", stream_index=" << lease.Index()
                                        << ", pool_size=" << cuda_stream_pool_->Size());
    return true;
}
#endif

bool Pipeline::ProcessBatch(
    const FrameBatch& batch,
    std::ofstream* result_file,
    IterationMetrics& metrics) {
    if (batch.frames.empty()) {
        EDGE_LOG_ERROR("Pipeline received an empty FrameBatch");
        return false;
    }

    std::ostringstream mapping;
    for (size_t i = 0; i < batch.frames.size(); ++i) {
        const auto& frame = batch.frames[i];
        mapping << " item=" << i
                << "(stream=" << frame.meta.stream_id
                << ",frame=" << frame.meta.frame_id << ")";
    }
    EDGE_LOG_INFO("Stage batch_scheduler done batch_id=" << batch.batch_id
                                                         << ", actual_batch_size="
                                                         << batch.ActualBatchSize()
                                                         << ", max_batch_size="
                                                         << config_.infer.batch_size
                                                         << ", trigger="
                                                         << batch.trigger_reason
                                                         << ", batch_wait_ms="
                                                         << batch.batch_wait_ms
                                                         << mapping.str());

    TensorBuffer input_tensor;
    std::vector<PreprocessMeta> preprocess_metas;
    Timer stage_timer;
    Timer process_timer;
    process_timer.Tic();

    double preprocess_ms = 0.0;
    double cpu_preprocess_ms = 0.0;
    double gpu_preprocess_ms = 0.0;
    double d2h_copy_ms = 0.0;
    double inference_ms = 0.0;
    PostprocessTiming postprocess_timing;
    std::vector<Detection> final_detections;
    {
        PROFILE_RANGE("pipeline_e2e");

        stage_timer.Tic();
        {
            PROFILE_RANGE("preprocess");
            if (config_.preprocess.type == "cpu") {
                PROFILE_RANGE("cpu_preprocess");
                if (!preprocessor_.RunBatch(batch.frames, input_tensor, preprocess_metas)) {
                    EDGE_LOG_ERROR("Stage preprocess failed for batch_id=" << batch.batch_id);
                    return false;
                }
                cpu_preprocess_ms = stage_timer.TocMs();
                preprocess_ms = cpu_preprocess_ms;
#ifdef EDGE_ENABLE_CUDA
            } else if (config_.preprocess.type == "gpu") {
                if (!gpu_preprocessor_) {
                    EDGE_LOG_ERROR("GpuPreprocessor was not initialized");
                    return false;
                }
                CudaStreamLease stream_lease;
                if (!AcquireCudaStream("gpu_preprocess", stream_lease)) {
                    return false;
                }

                GpuTensorBuffer gpu_tensor;
                {
                    PROFILE_RANGE("gpu_preprocess");
                    if (!gpu_preprocessor_->RunBatch(
                            batch.frames, stream_lease.Get(), gpu_tensor, preprocess_metas, &gpu_preprocess_ms)) {
                        return false;
                    }
                }
                {
                    PROFILE_RANGE("d2h_copy");
                    if (!gpu_preprocessor_->CopyToHost(gpu_tensor,
                                                       stream_lease.Get(),
                                                       input_tensor,
                                                       &d2h_copy_ms,
                                                       config_.cuda.enable_pinned_memory)) {
                        return false;
                    }
                }
                preprocess_ms = gpu_preprocess_ms + d2h_copy_ms;
#else
            } else if (config_.preprocess.type == "gpu") {
                EDGE_LOG_ERROR("preprocess.type=gpu requires -DENABLE_CUDA=ON");
                return false;
#endif
            } else {
                EDGE_LOG_ERROR("Unsupported preprocess.type: " << config_.preprocess.type);
                return false;
            }
        }
        EDGE_LOG_INFO("Stage preprocess done batch_id=" << batch.batch_id
                                                        << ", type="
                                                        << config_.preprocess.type
                                                        << ", actual_batch_size="
                                                        << batch.ActualBatchSize()
                                                        << ", preprocess_ms="
                                                        << preprocess_ms);

        TensorBuffer model_output;
        InferOutput infer_output;
        const bool use_trt_plugin_postprocess =
            config_.postprocess.decode_backend == "trt_plugin" && config_.postprocess.mode != "raw";
        stage_timer.Tic();
        {
            PROFILE_RANGE("inference");
            const bool infer_ok = use_trt_plugin_postprocess
                                      ? infer_engine_->Infer(input_tensor, infer_output)
                                      : infer_engine_->Infer(input_tensor, model_output);
            if (!infer_ok) {
                EDGE_LOG_ERROR("Stage inference failed for batch_id=" << batch.batch_id
                                                                      << ", engine="
                                                                      << infer_engine_->Name());
                return false;
            }
            if (use_trt_plugin_postprocess && infer_output.has_host_tensor) {
                model_output = std::move(infer_output.host_tensor);
            }
        }
        inference_ms = stage_timer.TocMs();
        const std::vector<int64_t>& output_shape =
            (use_trt_plugin_postprocess && infer_output.has_device_tensor)
                ? infer_output.device_tensor.shape
                : model_output.shape;
        EDGE_LOG_INFO("Stage inference done batch_id=" << batch.batch_id
                                                       << ", actual_batch_size="
                                                       << batch.ActualBatchSize()
                                                       << ", output_shape="
                                                       << ShapeToString(output_shape)
                                                       << ", device_output="
                                                       << (use_trt_plugin_postprocess &&
                                                           infer_output.has_device_tensor
                                                               ? "true"
                                                               : "false"));

        std::vector<FrameMeta> frame_metas;
        frame_metas.reserve(batch.frames.size());
        for (const auto& frame : batch.frames) {
            frame_metas.push_back(frame.meta);
        }

#ifdef EDGE_ENABLE_TENSORRT_PLUGIN
        if (use_trt_plugin_postprocess) {
            if (!trt_postprocess_engine_) {
                EDGE_LOG_ERROR("TrtPostprocessEngine was not initialized");
                return false;
            }
            CudaStreamLease trt_postprocess_stream;
            if (!AcquireCudaStream("trt_plugin_postprocess", trt_postprocess_stream)) {
                return false;
            }
            bool postprocess_ok = false;
            if (infer_output.has_device_tensor && infer_output.device_tensor.IsGpuFloat()) {
                postprocess_ok = trt_postprocess_engine_->RunDevice(infer_output.device_tensor,
                                                                    frame_metas,
                                                                    preprocess_metas,
                                                                    trt_postprocess_stream.Get(),
                                                                    final_detections,
                                                                    &postprocess_timing);
            } else if (infer_output.has_host_tensor) {
                EDGE_LOG_WARN("TensorRT plugin postprocess is using host fallback because "
                              "the inference backend did not expose a GPU output view.");
                postprocess_ok = trt_postprocess_engine_->Run(model_output,
                                                              frame_metas,
                                                              preprocess_metas,
                                                              trt_postprocess_stream.Get(),
                                                              final_detections,
                                                              &postprocess_timing);
            } else {
                EDGE_LOG_ERROR("TensorRT plugin postprocess requires either a GPU output view or host output tensor");
            }
            if (!postprocess_ok) {
                EDGE_LOG_ERROR("Stage trt_plugin postprocess failed for batch_id=" << batch.batch_id);
                return false;
            }
        } else
#endif
#ifdef EDGE_ENABLE_CUDA
        if (config_.postprocess.decode_backend == "gpu" && config_.postprocess.mode != "raw") {
            PROFILE_RANGE("postprocess");
            Timer postprocess_total_timer;
            postprocess_total_timer.Tic();
            std::vector<Detection> decoded_candidates;
            CudaStreamLease decode_stream;
            if (!AcquireCudaStream("gpu_postprocess", decode_stream)) {
                return false;
            }

            if (config_.postprocess.nms_backend == "gpu") {
                if (!gpu_decode_pre_nms_) {
                    EDGE_LOG_ERROR("GpuDecodePreNMS was not initialized");
                    return false;
                }
                if (!gpu_decode_pre_nms_->RunWithGpuNms(model_output,
                                                        frame_metas,
                                                        preprocess_metas,
                                                        config_.model.input_width,
                                                        config_.model.input_height,
                                                        decode_stream.Get(),
                                                        final_detections,
                                                        &postprocess_timing.gpu_decode_pre_nms_ms,
                                                        &postprocess_timing.gpu_nms_ms)) {
                    EDGE_LOG_ERROR("Stage gpu postprocess failed for batch_id=" << batch.batch_id);
                    return false;
                }
                postprocess_timing.decode_ms = postprocess_timing.gpu_decode_pre_nms_ms;
                postprocess_timing.nms_ms = postprocess_timing.gpu_nms_ms;
            } else {
                {
                    PROFILE_RANGE("decode");
                    if (!gpu_decode_pre_nms_) {
                        EDGE_LOG_ERROR("GpuDecodePreNMS was not initialized");
                        return false;
                    }
                    if (!gpu_decode_pre_nms_->Run(model_output,
                                                  frame_metas,
                                                  preprocess_metas,
                                                  config_.model.input_width,
                                                  config_.model.input_height,
                                                  decode_stream.Get(),
                                                  decoded_candidates,
                                                  &postprocess_timing.gpu_decode_pre_nms_ms)) {
                        EDGE_LOG_ERROR("Stage gpu_decode_pre_nms failed for batch_id=" << batch.batch_id);
                        return false;
                    }
                    postprocess_timing.decode_ms = postprocess_timing.gpu_decode_pre_nms_ms;
                }

                stage_timer.Tic();
                {
                    PROFILE_RANGE("nms");
                    final_detections = CPUPostprocessor::Nms(
                        std::move(decoded_candidates), config_.postprocess.nms_threshold, config_.postprocess.top_k);
                }
                postprocess_timing.nms_ms = stage_timer.TocMs();
                postprocess_timing.gpu_nms_ms = 0.0;
            }
            postprocess_timing.total_ms = postprocess_total_timer.TocMs();
            EDGE_LOG_INFO("[Postprocess] decode_backend=gpu"
                          << ", nms_backend=" << config_.postprocess.nms_backend
                          << ", gpu_decode_pre_nms_ms=" << postprocess_timing.gpu_decode_pre_nms_ms
                          << ", gpu_nms_ms=" << postprocess_timing.gpu_nms_ms
                          << ", nms_latency_ms=" << postprocess_timing.nms_ms
                          << ", postprocess_latency_ms=" << postprocess_timing.total_ms
                          << ", detection_count=" << final_detections.size());
        } else
#endif
        {
            if (!postprocessor_.Run(model_output,
                                    frame_metas,
                                    preprocess_metas,
                                    final_detections,
                                    &postprocess_timing)) {
                EDGE_LOG_ERROR("Stage postprocess failed for batch_id=" << batch.batch_id
                                                                        << ", output_shape="
                                                                        << ShapeToString(model_output.shape));
                return false;
            }
        }
        EDGE_LOG_INFO("Stage postprocess done batch_id=" << batch.batch_id
                                                         << ", actual_batch_size="
                                                         << batch.ActualBatchSize());
    }
    const double processing_ms = process_timer.TocMs();

    std::map<std::pair<int, int>, int> detection_counts;
    for (const auto& det : final_detections) {
        ++detection_counts[{det.stream_id, det.frame_id}];
    }

    for (const auto& frame : batch.frames) {
        const auto key = std::make_pair(frame.meta.stream_id, frame.meta.frame_id);
        const int detection_count = detection_counts.count(key) > 0 ? detection_counts[key] : 0;
        EDGE_LOG_INFO("[BatchOutput] batch_id=" << batch.batch_id
                                                << ", stream_id=" << frame.meta.stream_id
                                                << ", frame_id=" << frame.meta.frame_id
                                                << ", detection_count="
                                                << detection_count);
    }

    for (const auto& det : final_detections) {
        EDGE_LOG_INFO("Detection stream=" << det.stream_id
                                          << ", frame=" << det.frame_id
                                          << ", class=" << det.class_id
                                          << ", score=" << det.score
                                          << ", box=(" << det.x1 << ", " << det.y1
                                          << ", " << det.x2 << ", " << det.y2 << ")");
        if (result_file != nullptr) {
            *result_file << det.stream_id << ','
                         << det.frame_id << ','
                         << det.class_id << ','
                         << det.score << ','
                         << det.x1 << ','
                         << det.y1 << ','
                         << det.x2 << ','
                         << det.y2 << '\n';
        }
    }

    metrics.num_streams = config_.input.num_streams;
    metrics.batch_size = config_.infer.batch_size;
    metrics.actual_batch_size = batch.ActualBatchSize();
    metrics.predictor_pool_size = 1;
    metrics.predictor_worker_id = -1;
    metrics.batch_wait_ms = batch.batch_wait_ms;
    metrics.batch_latency_ms = batch.batch_wait_ms + processing_ms;
    metrics.inference_queue_wait_ms = 0.0;
    metrics.video_decode_ms = batch.video_decode_ms;
    metrics.preprocess_backend = config_.preprocess.type;
    metrics.preprocess_ms = preprocess_ms;
    metrics.cpu_preprocess_ms = cpu_preprocess_ms;
    metrics.gpu_preprocess_ms = gpu_preprocess_ms;
    metrics.d2h_copy_ms = d2h_copy_ms;
    metrics.inference_ms = inference_ms;
    metrics.decode_backend = config_.postprocess.mode == "raw" ? "raw" : config_.postprocess.decode_backend;
    metrics.nms_backend = config_.postprocess.mode == "raw" ? "raw" : config_.postprocess.nms_backend;
    metrics.decode_ms = postprocess_timing.decode_ms;
    metrics.cpu_decode_ms = postprocess_timing.cpu_decode_ms;
    metrics.gpu_decode_pre_nms_ms = postprocess_timing.gpu_decode_pre_nms_ms;
    metrics.nms_ms = postprocess_timing.nms_ms;
    metrics.gpu_nms_ms = postprocess_timing.gpu_nms_ms;
    metrics.trt_plugin_ms = postprocess_timing.trt_plugin_ms;
    metrics.postprocess_ms = postprocess_timing.total_ms;
    metrics.e2e_ms = metrics.video_decode_ms + metrics.batch_latency_ms;
    metrics.fps = metrics.e2e_ms > 0.0
                      ? static_cast<double>(metrics.actual_batch_size) * 1000.0 / metrics.e2e_ms
                      : 0.0;

    if (config_.profile.enable_timer) {
        EDGE_LOG_INFO("[Profile] batch_id=" << batch.batch_id
                                            << ", actual_batch_size="
                                            << metrics.actual_batch_size
                                            << ", batch_wait_ms="
                                            << metrics.batch_wait_ms
                                            << ", batch_latency_ms="
                                            << metrics.batch_latency_ms
                                            << ", predictor_pool_size="
                                            << metrics.predictor_pool_size
                                            << ", predictor_worker_id="
                                            << metrics.predictor_worker_id
                                            << ", inference_queue_wait_ms="
                                            << metrics.inference_queue_wait_ms
                                            << ", video_decode_ms="
                                            << metrics.video_decode_ms
                                            << ", preprocess_backend="
                                            << metrics.preprocess_backend
                                            << ", preprocess_ms="
                                            << metrics.preprocess_ms
                                            << ", cpu_preprocess_ms="
                                            << metrics.cpu_preprocess_ms
                                            << ", gpu_preprocess_ms="
                                            << metrics.gpu_preprocess_ms
                                            << ", d2h_copy_ms="
                                            << metrics.d2h_copy_ms
                                            << ", inference_ms="
                                            << metrics.inference_ms
                                            << ", decode_backend="
                                            << metrics.decode_backend
                                            << ", nms_backend="
                                            << metrics.nms_backend
                                            << ", decode_ms="
                                            << metrics.decode_ms
                                            << ", cpu_decode_ms="
                                            << metrics.cpu_decode_ms
                                            << ", gpu_decode_pre_nms_ms="
                                            << metrics.gpu_decode_pre_nms_ms
                                            << ", nms_ms="
                                            << metrics.nms_ms
                                            << ", gpu_nms_ms="
                                            << metrics.gpu_nms_ms
                                            << ", trt_plugin_ms="
                                            << metrics.trt_plugin_ms
                                            << ", postprocess_ms="
                                            << metrics.postprocess_ms
                                            << ", e2e_ms="
                                            << metrics.e2e_ms
                                            << ", fps="
                                            << metrics.fps);
    }

    return true;
}

bool Pipeline::RunWithPredictorPool() {
    if (!predictor_pool_) {
        EDGE_LOG_ERROR("Pipeline predictor pool path requested before PredictorPool init");
        return false;
    }

    EDGE_LOG_INFO("Pipeline run started with " << predictor_pool_->Name());
    int processed_frames = 0;
    int processed_batches = 0;
    std::ofstream result_file;

    if (config_.output.save_result) {
        result_file.open(config_.output.result_path);
        if (!result_file.is_open()) {
            EDGE_LOG_ERROR("Failed to open output.result_path: " << config_.output.result_path);
            return false;
        }
        EDGE_LOG_INFO("Saving mock results to " << config_.output.result_path);
    }

    std::vector<bool> active(sources_.size(), true);
    int active_count = static_cast<int>(sources_.size());
    size_t next_source_index = 0;
    int seen_iters = 0;
    int benchmark_iters = 0;
    bool reached_benchmark_limit = false;
    int outstanding = 0;
    std::map<int, PreparedBatch> prepared_batches;

    const bool use_trt_plugin_postprocess =
        config_.postprocess.decode_backend == "trt_plugin" && config_.postprocess.mode != "raw";

    auto finalize_one_result = [&]() -> bool {
        PredictorPool::Result result;
        if (!predictor_pool_->Pop(result)) {
            EDGE_LOG_ERROR("PredictorPool returned no result while outstanding=" << outstanding);
            return false;
        }

        auto prepared_it = prepared_batches.find(result.batch_id);
        if (prepared_it == prepared_batches.end()) {
            predictor_pool_->Release(result);
            EDGE_LOG_ERROR("PredictorPool result references unknown batch_id=" << result.batch_id);
            return false;
        }

        PreparedBatch prepared = std::move(prepared_it->second);
        prepared_batches.erase(prepared_it);
        --outstanding;

        if (!result.ok) {
            predictor_pool_->Release(result);
            EDGE_LOG_ERROR("Stage inference failed for batch_id=" << result.batch_id
                                                                  << ", worker_id="
                                                                  << result.worker_id
                                                                  << ", error="
                                                                  << result.error);
            return false;
        }

        IterationMetrics metrics;
        const TensorBuffer* host_output = result.used_infer_output ? nullptr : &result.host_output;
        const InferOutput* infer_output = result.used_infer_output ? &result.infer_output : nullptr;
        if (!PostprocessPreparedBatch(prepared,
                                      host_output,
                                      infer_output,
                                      result.inference_ms,
                                      result.queue_wait_ms,
                                      result.worker_id,
                                      result_file.is_open() ? &result_file : nullptr,
                                      metrics)) {
            predictor_pool_->Release(result);
            return false;
        }
        predictor_pool_->Release(result);

        processed_frames += prepared.batch.ActualBatchSize();
        ++processed_batches;

        const bool warmup = seen_iters < config_.benchmark.warmup_iters;
        ++seen_iters;
        if (warmup) {
            EDGE_LOG_INFO("[Benchmark] warmup iter=" << seen_iters
                                                    << ", batch_id=" << prepared.batch.batch_id
                                                    << ", actual_batch_size="
                                                    << prepared.batch.ActualBatchSize()
                                                    << ", worker_id=" << result.worker_id
                                                    << ", e2e_ms=" << metrics.e2e_ms);
            return true;
        }

        ++benchmark_iters;
        metrics.iter = benchmark_iters;
        profiler_.AddIteration(metrics);
        if (benchmark_iters == 1 || benchmark_iters % 10 == 0) {
            profiler_.ReportToStdout(std::cout);
        }

        if (config_.benchmark.benchmark_iters > 0 &&
            benchmark_iters >= config_.benchmark.benchmark_iters) {
            reached_benchmark_limit = true;
        }
        return true;
    };

    auto submit_batch = [&](const FrameBatch& batch) -> bool {
        PreparedBatch prepared;
        if (!PreprocessBatch(batch, prepared)) {
            return false;
        }

        PredictorPool::Request request;
        request.batch_id = prepared.batch.batch_id;
        request.prefer_device_output = use_trt_plugin_postprocess;
        request.input = std::move(prepared.input_tensor);
        prepared_batches[prepared.batch.batch_id] = std::move(prepared);
        if (!predictor_pool_->Submit(std::move(request))) {
            prepared_batches.erase(batch.batch_id);
            return false;
        }
        ++outstanding;
        EDGE_LOG_INFO("[PredictorPool] submitted batch_id=" << batch.batch_id
                                                            << ", outstanding="
                                                            << outstanding);
        return true;
    };

    while ((active_count > 0 || batch_scheduler_.HasPending() || outstanding > 0) &&
           !(reached_benchmark_limit && outstanding == 0)) {
        if ((reached_benchmark_limit || outstanding >= predictor_pool_->Size()) && outstanding > 0) {
            if (!finalize_one_result()) {
                return false;
            }
            continue;
        }

        FrameBatch ready_batch;
        if (!reached_benchmark_limit && batch_scheduler_.PopReadyBatch(ready_batch)) {
            if (!submit_batch(ready_batch)) {
                return false;
            }
            continue;
        }

        if (!reached_benchmark_limit && active_count <= 0) {
            if (batch_scheduler_.Flush(ready_batch)) {
                if (!submit_batch(ready_batch)) {
                    return false;
                }
                continue;
            }
            if (outstanding > 0) {
                if (!finalize_one_result()) {
                    return false;
                }
            }
            continue;
        }

        bool read_frame = false;
        if (!reached_benchmark_limit) {
            for (size_t scan = 0; scan < sources_.size(); ++scan) {
                const size_t index = next_source_index;
                next_source_index = (next_source_index + 1) % sources_.size();
                if (!active[index]) {
                    continue;
                }

                VideoFrame frame;
                Timer video_decode_timer;
                video_decode_timer.Tic();
                bool read_ok = false;
                {
                    PROFILE_RANGE("video_decode");
                    read_ok = sources_[index]->Read(frame);
                }
                const double video_decode_ms = video_decode_timer.TocMs();
                if (!read_ok) {
                    active[index] = false;
                    --active_count;
                    EDGE_LOG_INFO("VideoSource exhausted stream_id=" << sources_[index]->StreamId());
                    continue;
                }

                EDGE_LOG_INFO("Stage video_source done stream_id=" << frame.meta.stream_id
                                                                   << ", frame_id="
                                                                   << frame.meta.frame_id
                                                                   << ", timestamp_ms="
                                                                   << frame.meta.timestamp_ms);
                batch_scheduler_.Enqueue(std::move(frame), video_decode_ms);
                read_frame = true;
                break;
            }
        }

        if (!read_frame && outstanding > 0) {
            if (!finalize_one_result()) {
                return false;
            }
        }
    }

    EDGE_LOG_INFO("Pipeline run finished, processed_frames=" << processed_frames
                                                             << ", processed_batches="
                                                             << processed_batches);
    if (profiler_.Count() > 0) {
        profiler_.ReportToStdout(std::cout);
        if (!profiler_.SaveCsv(config_.benchmark.output_csv)) {
            return false;
        }
    } else {
        EDGE_LOG_WARN("No benchmark samples were collected; CSV will not be written");
    }
    return processed_frames > 0;
}

bool Pipeline::PreprocessBatch(const FrameBatch& batch, Pipeline::PreparedBatch& prepared) {
    if (batch.frames.empty()) {
        EDGE_LOG_ERROR("Pipeline received an empty FrameBatch");
        return false;
    }

    prepared = PreparedBatch{};
    prepared.batch = batch;
    prepared.process_start_time = std::chrono::steady_clock::now();
    prepared.frame_metas.reserve(batch.frames.size());
    for (const auto& frame : batch.frames) {
        prepared.frame_metas.push_back(frame.meta);
    }

    std::ostringstream mapping;
    for (size_t i = 0; i < batch.frames.size(); ++i) {
        const auto& frame = batch.frames[i];
        mapping << " item=" << i
                << "(stream=" << frame.meta.stream_id
                << ",frame=" << frame.meta.frame_id << ")";
    }
    EDGE_LOG_INFO("Stage batch_scheduler done batch_id=" << batch.batch_id
                                                         << ", actual_batch_size="
                                                         << batch.ActualBatchSize()
                                                         << ", max_batch_size="
                                                         << config_.infer.batch_size
                                                         << ", trigger="
                                                         << batch.trigger_reason
                                                         << ", batch_wait_ms="
                                                         << batch.batch_wait_ms
                                                         << mapping.str());

    Timer stage_timer;
    stage_timer.Tic();
    {
        PROFILE_RANGE("preprocess");
        if (config_.preprocess.type == "cpu") {
            PROFILE_RANGE("cpu_preprocess");
            if (!preprocessor_.RunBatch(batch.frames, prepared.input_tensor, prepared.preprocess_metas)) {
                EDGE_LOG_ERROR("Stage preprocess failed for batch_id=" << batch.batch_id);
                return false;
            }
            prepared.cpu_preprocess_ms = stage_timer.TocMs();
            prepared.preprocess_ms = prepared.cpu_preprocess_ms;
#ifdef EDGE_ENABLE_CUDA
        } else if (config_.preprocess.type == "gpu") {
            if (!gpu_preprocessor_) {
                EDGE_LOG_ERROR("GpuPreprocessor was not initialized");
                return false;
            }
            CudaStreamLease stream;
            if (!AcquireCudaStream("gpu_preprocess", stream)) {
                return false;
            }

            GpuTensorBuffer gpu_tensor;
            {
                PROFILE_RANGE("gpu_preprocess");
                if (!gpu_preprocessor_->RunBatch(
                        batch.frames, stream.Get(), gpu_tensor, prepared.preprocess_metas, &prepared.gpu_preprocess_ms)) {
                    return false;
                }
            }
            {
                PROFILE_RANGE("d2h_copy");
                if (!gpu_preprocessor_->CopyToHost(
                        gpu_tensor,
                        stream.Get(),
                        prepared.input_tensor,
                        &prepared.d2h_copy_ms,
                        config_.cuda.enable_pinned_memory)) {
                    return false;
                }
            }
            prepared.preprocess_ms = prepared.gpu_preprocess_ms + prepared.d2h_copy_ms;
#else
        } else if (config_.preprocess.type == "gpu") {
            EDGE_LOG_ERROR("preprocess.type=gpu requires -DENABLE_CUDA=ON");
            return false;
#endif
        } else {
            EDGE_LOG_ERROR("Unsupported preprocess.type: " << config_.preprocess.type);
            return false;
        }
    }

    EDGE_LOG_INFO("Stage preprocess done batch_id=" << batch.batch_id
                                                    << ", type="
                                                    << config_.preprocess.type
                                                    << ", actual_batch_size="
                                                    << batch.ActualBatchSize()
                                                    << ", preprocess_ms="
                                                    << prepared.preprocess_ms);
    return true;
}

bool Pipeline::PostprocessPreparedBatch(
    const Pipeline::PreparedBatch& prepared,
    const TensorBuffer* host_output,
    const InferOutput* infer_output,
    double inference_ms,
    double inference_queue_wait_ms,
    int predictor_worker_id,
    std::ofstream* result_file,
    IterationMetrics& metrics) {
    PostprocessTiming postprocess_timing;
    std::vector<Detection> final_detections;
    const bool use_trt_plugin_postprocess =
        config_.postprocess.decode_backend == "trt_plugin" && config_.postprocess.mode != "raw";

    const TensorBuffer* effective_host_output = host_output;
    if (infer_output != nullptr && infer_output->has_host_tensor) {
        effective_host_output = &infer_output->host_tensor;
    }

#ifdef EDGE_ENABLE_TENSORRT_PLUGIN
    if (use_trt_plugin_postprocess) {
        if (!trt_postprocess_engine_) {
            EDGE_LOG_ERROR("TrtPostprocessEngine was not initialized");
            return false;
        }
        CudaStreamLease trt_postprocess_stream;
        if (!AcquireCudaStream("trt_plugin_postprocess", trt_postprocess_stream)) {
            return false;
        }
        bool postprocess_ok = false;
        if (infer_output != nullptr &&
            infer_output->has_device_tensor &&
            infer_output->device_tensor.IsGpuFloat()) {
            postprocess_ok = trt_postprocess_engine_->RunDevice(infer_output->device_tensor,
                                                                prepared.frame_metas,
                                                                prepared.preprocess_metas,
                                                                trt_postprocess_stream.Get(),
                                                                final_detections,
                                                                &postprocess_timing);
        } else if (effective_host_output != nullptr) {
            EDGE_LOG_WARN("TensorRT plugin postprocess is using host fallback because "
                          "the inference backend did not expose a GPU output view.");
            postprocess_ok = trt_postprocess_engine_->Run(*effective_host_output,
                                                          prepared.frame_metas,
                                                          prepared.preprocess_metas,
                                                          trt_postprocess_stream.Get(),
                                                          final_detections,
                                                          &postprocess_timing);
        } else {
            EDGE_LOG_ERROR("TensorRT plugin postprocess requires either a GPU output view or host output tensor");
        }
        if (!postprocess_ok) {
            EDGE_LOG_ERROR("Stage trt_plugin postprocess failed for batch_id="
                           << prepared.batch.batch_id);
            return false;
        }
    } else
#endif
#ifdef EDGE_ENABLE_CUDA
    if (config_.postprocess.decode_backend == "gpu" && config_.postprocess.mode != "raw") {
        if (effective_host_output == nullptr) {
            EDGE_LOG_ERROR("GPU decode path requires a host model output tensor");
            return false;
        }
        PROFILE_RANGE("postprocess");
        Timer postprocess_total_timer;
        Timer stage_timer;
        postprocess_total_timer.Tic();
        std::vector<Detection> decoded_candidates;
        CudaStreamLease decode_stream;
        if (!AcquireCudaStream("gpu_postprocess", decode_stream)) {
            return false;
        }

        if (config_.postprocess.nms_backend == "gpu") {
            if (!gpu_decode_pre_nms_) {
                EDGE_LOG_ERROR("GpuDecodePreNMS was not initialized");
                return false;
            }
            if (!gpu_decode_pre_nms_->RunWithGpuNms(*effective_host_output,
                                                    prepared.frame_metas,
                                                    prepared.preprocess_metas,
                                                    config_.model.input_width,
                                                    config_.model.input_height,
                                                    decode_stream.Get(),
                                                    final_detections,
                                                    &postprocess_timing.gpu_decode_pre_nms_ms,
                                                    &postprocess_timing.gpu_nms_ms)) {
                EDGE_LOG_ERROR("Stage gpu postprocess failed for batch_id="
                               << prepared.batch.batch_id);
                return false;
            }
            postprocess_timing.decode_ms = postprocess_timing.gpu_decode_pre_nms_ms;
            postprocess_timing.nms_ms = postprocess_timing.gpu_nms_ms;
        } else {
            {
                PROFILE_RANGE("decode");
                if (!gpu_decode_pre_nms_) {
                    EDGE_LOG_ERROR("GpuDecodePreNMS was not initialized");
                    return false;
                }
                if (!gpu_decode_pre_nms_->Run(*effective_host_output,
                                              prepared.frame_metas,
                                              prepared.preprocess_metas,
                                              config_.model.input_width,
                                              config_.model.input_height,
                                              decode_stream.Get(),
                                              decoded_candidates,
                                              &postprocess_timing.gpu_decode_pre_nms_ms)) {
                    EDGE_LOG_ERROR("Stage gpu_decode_pre_nms failed for batch_id="
                                   << prepared.batch.batch_id);
                    return false;
                }
                postprocess_timing.decode_ms = postprocess_timing.gpu_decode_pre_nms_ms;
            }

            stage_timer.Tic();
            {
                PROFILE_RANGE("nms");
                final_detections = CPUPostprocessor::Nms(
                    std::move(decoded_candidates), config_.postprocess.nms_threshold, config_.postprocess.top_k);
            }
            postprocess_timing.nms_ms = stage_timer.TocMs();
            postprocess_timing.gpu_nms_ms = 0.0;
        }
        postprocess_timing.total_ms = postprocess_total_timer.TocMs();
        EDGE_LOG_INFO("[Postprocess] decode_backend=gpu"
                      << ", nms_backend=" << config_.postprocess.nms_backend
                      << ", gpu_decode_pre_nms_ms=" << postprocess_timing.gpu_decode_pre_nms_ms
                      << ", gpu_nms_ms=" << postprocess_timing.gpu_nms_ms
                      << ", nms_latency_ms=" << postprocess_timing.nms_ms
                      << ", postprocess_latency_ms=" << postprocess_timing.total_ms
                      << ", detection_count=" << final_detections.size());
    } else
#endif
    {
        if (effective_host_output == nullptr) {
            EDGE_LOG_ERROR("CPU postprocess path requires a host model output tensor");
            return false;
        }
        if (!postprocessor_.Run(*effective_host_output,
                                prepared.frame_metas,
                                prepared.preprocess_metas,
                                final_detections,
                                &postprocess_timing)) {
            EDGE_LOG_ERROR("Stage postprocess failed for batch_id=" << prepared.batch.batch_id
                                                                    << ", output_shape="
                                                                    << ShapeToString(effective_host_output->shape));
            return false;
        }
    }

    EDGE_LOG_INFO("Stage postprocess done batch_id=" << prepared.batch.batch_id
                                                     << ", actual_batch_size="
                                                     << prepared.batch.ActualBatchSize()
                                                     << ", predictor_worker_id="
                                                     << predictor_worker_id);

    std::map<std::pair<int, int>, int> detection_counts;
    for (const auto& det : final_detections) {
        ++detection_counts[{det.stream_id, det.frame_id}];
    }

    for (const auto& frame : prepared.batch.frames) {
        const auto key = std::make_pair(frame.meta.stream_id, frame.meta.frame_id);
        const int detection_count = detection_counts.count(key) > 0 ? detection_counts[key] : 0;
        EDGE_LOG_INFO("[BatchOutput] batch_id=" << prepared.batch.batch_id
                                                << ", stream_id=" << frame.meta.stream_id
                                                << ", frame_id=" << frame.meta.frame_id
                                                << ", detection_count="
                                                << detection_count);
    }

    for (const auto& det : final_detections) {
        EDGE_LOG_INFO("Detection stream=" << det.stream_id
                                          << ", frame=" << det.frame_id
                                          << ", class=" << det.class_id
                                          << ", score=" << det.score
                                          << ", box=(" << det.x1 << ", " << det.y1
                                          << ", " << det.x2 << ", " << det.y2 << ")");
        if (result_file != nullptr) {
            *result_file << det.stream_id << ','
                         << det.frame_id << ','
                         << det.class_id << ','
                         << det.score << ','
                         << det.x1 << ','
                         << det.y1 << ','
                         << det.x2 << ','
                         << det.y2 << '\n';
        }
    }

    const auto process_end_time = std::chrono::steady_clock::now();
    const auto process_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            process_end_time - prepared.process_start_time).count();
    const double processing_ms = static_cast<double>(process_us) / 1000.0;

    metrics.num_streams = config_.input.num_streams;
    metrics.batch_size = config_.infer.batch_size;
    metrics.actual_batch_size = prepared.batch.ActualBatchSize();
    metrics.predictor_pool_size = config_.infer.predictor_pool_size;
    metrics.predictor_worker_id = predictor_worker_id;
    metrics.batch_wait_ms = prepared.batch.batch_wait_ms;
    metrics.batch_latency_ms = prepared.batch.batch_wait_ms + processing_ms;
    metrics.inference_queue_wait_ms = inference_queue_wait_ms;
    metrics.video_decode_ms = prepared.batch.video_decode_ms;
    metrics.preprocess_backend = config_.preprocess.type;
    metrics.preprocess_ms = prepared.preprocess_ms;
    metrics.cpu_preprocess_ms = prepared.cpu_preprocess_ms;
    metrics.gpu_preprocess_ms = prepared.gpu_preprocess_ms;
    metrics.d2h_copy_ms = prepared.d2h_copy_ms;
    metrics.inference_ms = inference_ms;
    metrics.decode_backend = config_.postprocess.mode == "raw" ? "raw" : config_.postprocess.decode_backend;
    metrics.nms_backend = config_.postprocess.mode == "raw" ? "raw" : config_.postprocess.nms_backend;
    metrics.decode_ms = postprocess_timing.decode_ms;
    metrics.cpu_decode_ms = postprocess_timing.cpu_decode_ms;
    metrics.gpu_decode_pre_nms_ms = postprocess_timing.gpu_decode_pre_nms_ms;
    metrics.nms_ms = postprocess_timing.nms_ms;
    metrics.gpu_nms_ms = postprocess_timing.gpu_nms_ms;
    metrics.trt_plugin_ms = postprocess_timing.trt_plugin_ms;
    metrics.postprocess_ms = postprocess_timing.total_ms;
    metrics.e2e_ms = metrics.video_decode_ms + metrics.batch_latency_ms;
    metrics.fps = metrics.e2e_ms > 0.0
                      ? static_cast<double>(metrics.actual_batch_size) * 1000.0 / metrics.e2e_ms
                      : 0.0;

    if (config_.profile.enable_timer) {
        EDGE_LOG_INFO("[Profile] batch_id=" << prepared.batch.batch_id
                                            << ", actual_batch_size="
                                            << metrics.actual_batch_size
                                            << ", predictor_pool_size="
                                            << metrics.predictor_pool_size
                                            << ", predictor_worker_id="
                                            << metrics.predictor_worker_id
                                            << ", batch_wait_ms="
                                            << metrics.batch_wait_ms
                                            << ", batch_latency_ms="
                                            << metrics.batch_latency_ms
                                            << ", inference_queue_wait_ms="
                                            << metrics.inference_queue_wait_ms
                                            << ", video_decode_ms="
                                            << metrics.video_decode_ms
                                            << ", preprocess_backend="
                                            << metrics.preprocess_backend
                                            << ", preprocess_ms="
                                            << metrics.preprocess_ms
                                            << ", inference_ms="
                                            << metrics.inference_ms
                                            << ", decode_backend="
                                            << metrics.decode_backend
                                            << ", nms_backend="
                                            << metrics.nms_backend
                                            << ", postprocess_ms="
                                            << metrics.postprocess_ms
                                            << ", e2e_ms="
                                            << metrics.e2e_ms
                                            << ", fps="
                                            << metrics.fps);
    }

    return true;
}

}  // namespace edge
