#include "profiling/profiler.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "common/logging.h"

namespace edge {

void Timer::Tic() {
    start_ = std::chrono::steady_clock::now();
}

double Timer::TocMs() const {
    const auto end = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    return static_cast<double>(us) / 1000.0;
}

void AverageMeter::Add(double value) {
    ++count_;
    sum_ += value;
    min_ = std::min(min_, value);
    max_ = std::max(max_, value);
}

int AverageMeter::Count() const {
    return count_;
}

double AverageMeter::Average() const {
    return count_ > 0 ? sum_ / static_cast<double>(count_) : 0.0;
}

double AverageMeter::Min() const {
    return count_ > 0 ? min_ : 0.0;
}

double AverageMeter::Max() const {
    return count_ > 0 ? max_ : 0.0;
}

void PercentileMeter::Add(double value) {
    samples_.push_back(value);
}

int PercentileMeter::Count() const {
    return static_cast<int>(samples_.size());
}

double PercentileMeter::Average() const {
    if (samples_.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(samples_.begin(), samples_.end(), 0.0);
    return sum / static_cast<double>(samples_.size());
}

double PercentileMeter::Min() const {
    if (samples_.empty()) {
        return 0.0;
    }
    return *std::min_element(samples_.begin(), samples_.end());
}

double PercentileMeter::Max() const {
    if (samples_.empty()) {
        return 0.0;
    }
    return *std::max_element(samples_.begin(), samples_.end());
}

double PercentileMeter::Percentile(double percentile) const {
    if (samples_.empty()) {
        return 0.0;
    }

    std::vector<double> sorted = samples_;
    std::sort(sorted.begin(), sorted.end());
    const double clamped = std::clamp(percentile, 0.0, 100.0);
    const double rank = (clamped / 100.0) * static_cast<double>(sorted.size() - 1);
    const auto index = static_cast<size_t>(rank + 0.5);
    return sorted[index];
}

void Profiler::AddIteration(const IterationMetrics& metrics) {
    iterations_.push_back(metrics);
    ++actual_batch_size_counts_[metrics.actual_batch_size];
    AddStage("batch_wait", metrics.batch_wait_ms);
    AddStage("batch_latency", metrics.batch_latency_ms);
    if (metrics.inference_queue_wait_ms > 0.0) {
        AddStage("inference_queue_wait", metrics.inference_queue_wait_ms);
    }
    AddStage("video_decode", metrics.video_decode_ms);
    AddStage("preprocess", metrics.preprocess_ms);
    if (metrics.cpu_preprocess_ms > 0.0) {
        AddStage("cpu_preprocess", metrics.cpu_preprocess_ms);
    }
    if (metrics.gpu_preprocess_ms > 0.0) {
        AddStage("gpu_preprocess", metrics.gpu_preprocess_ms);
    }
    if (metrics.d2h_copy_ms > 0.0) {
        AddStage("d2h_copy", metrics.d2h_copy_ms);
    }
    AddStage("inference", metrics.inference_ms);
    AddStage("decode", metrics.decode_ms);
    if (metrics.cpu_decode_ms > 0.0) {
        AddStage("cpu_decode", metrics.cpu_decode_ms);
    }
    if (metrics.gpu_decode_pre_nms_ms > 0.0) {
        AddStage("gpu_decode_pre_nms", metrics.gpu_decode_pre_nms_ms);
    }
    AddStage("nms", metrics.nms_ms);
    if (metrics.gpu_nms_ms > 0.0) {
        AddStage("gpu_nms", metrics.gpu_nms_ms);
    }
    if (metrics.trt_plugin_ms > 0.0) {
        AddStage("trt_plugin_postprocess", metrics.trt_plugin_ms);
    }
    AddStage("postprocess", metrics.postprocess_ms);
    AddStage("e2e", metrics.e2e_ms);
}

int Profiler::Count() const {
    return static_cast<int>(iterations_.size());
}

void Profiler::ReportToStdout(std::ostream& os) const {
    os << "[Profiler] samples=" << iterations_.size() << '\n';
    if (!actual_batch_size_counts_.empty()) {
        os << "[Profiler] actual_batch_size_distribution";
        for (const auto& item : actual_batch_size_counts_) {
            os << " size=" << item.first << ":count=" << item.second;
        }
        os << '\n';
    }
    for (const auto& item : stage_meters_) {
        const PercentileMeter& meter = item.second;
        os << "[Profiler] " << item.first
           << " avg=" << meter.Average()
           << " min=" << meter.Min()
           << " max=" << meter.Max()
           << " p50=" << meter.Percentile(50.0)
           << " p90=" << meter.Percentile(90.0)
           << " p99=" << meter.Percentile(99.0)
           << " ms\n";
    }
}

bool Profiler::SaveCsv(const std::string& path) const {
    std::ofstream output(path);
    if (!output.is_open()) {
        EDGE_LOG_ERROR("Failed to open benchmark CSV for writing: " << path);
        return false;
    }

    output << "iter,num_streams,batch_size,actual_batch_size,predictor_pool_size,"
           << "predictor_worker_id,preprocess_backend,"
           << "batch_wait_ms,batch_latency_ms,inference_queue_wait_ms,video_decode_ms,preprocess_ms,"
           << "cpu_preprocess_ms,gpu_preprocess_ms,d2h_copy_ms,"
           << "inference_ms,decode_backend,nms_backend,decode_ms,cpu_decode_ms,gpu_decode_pre_nms_ms,"
           << "nms_ms,gpu_nms_ms,trt_plugin_ms,postprocess_ms,e2e_ms,fps\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& metrics : iterations_) {
        output << metrics.iter << ','
               << metrics.num_streams << ','
               << metrics.batch_size << ','
               << metrics.actual_batch_size << ','
               << metrics.predictor_pool_size << ','
               << metrics.predictor_worker_id << ','
               << metrics.preprocess_backend << ','
               << metrics.batch_wait_ms << ','
               << metrics.batch_latency_ms << ','
               << metrics.inference_queue_wait_ms << ','
               << metrics.video_decode_ms << ','
               << metrics.preprocess_ms << ','
               << metrics.cpu_preprocess_ms << ','
               << metrics.gpu_preprocess_ms << ','
               << metrics.d2h_copy_ms << ','
               << metrics.inference_ms << ','
               << metrics.decode_backend << ','
               << metrics.nms_backend << ','
               << metrics.decode_ms << ','
               << metrics.cpu_decode_ms << ','
               << metrics.gpu_decode_pre_nms_ms << ','
               << metrics.nms_ms << ','
               << metrics.gpu_nms_ms << ','
               << metrics.trt_plugin_ms << ','
               << metrics.postprocess_ms << ','
               << metrics.e2e_ms << ','
               << metrics.fps << '\n';
    }

    EDGE_LOG_INFO("Saved benchmark CSV: " << path << ", rows=" << iterations_.size());
    return true;
}

void Profiler::AddStage(const std::string& name, double value) {
    stage_meters_[name].Add(value);
}

}  // namespace edge
