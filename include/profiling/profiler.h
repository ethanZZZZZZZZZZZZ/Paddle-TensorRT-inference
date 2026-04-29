#pragma once

#include <chrono>
#include <iosfwd>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace edge {

class Timer {
public:
    void Tic();
    double TocMs() const;

private:
    std::chrono::steady_clock::time_point start_;
};

class AverageMeter {
public:
    void Add(double value);
    int Count() const;
    double Average() const;
    double Min() const;
    double Max() const;

private:
    int count_ = 0;
    double sum_ = 0.0;
    double min_ = std::numeric_limits<double>::max();
    double max_ = std::numeric_limits<double>::lowest();
};

class PercentileMeter {
public:
    void Add(double value);
    int Count() const;
    double Average() const;
    double Min() const;
    double Max() const;
    double Percentile(double percentile) const;

private:
    std::vector<double> samples_;
};

struct IterationMetrics {
    int iter = 0;
    int num_streams = 1;
    int batch_size = 1;
    int actual_batch_size = 1;
    int predictor_pool_size = 1;
    int predictor_worker_id = -1;
    std::string preprocess_backend = "cpu";
    double batch_wait_ms = 0.0;
    double batch_latency_ms = 0.0;
    double inference_queue_wait_ms = 0.0;
    double video_decode_ms = 0.0;
    double preprocess_ms = 0.0;
    double cpu_preprocess_ms = 0.0;
    double gpu_preprocess_ms = 0.0;
    double d2h_copy_ms = 0.0;
    double inference_ms = 0.0;
    std::string decode_backend = "cpu";
    std::string nms_backend = "cpu";
    double decode_ms = 0.0;
    double cpu_decode_ms = 0.0;
    double gpu_decode_pre_nms_ms = 0.0;
    double nms_ms = 0.0;
    double gpu_nms_ms = 0.0;
    double trt_plugin_ms = 0.0;
    double postprocess_ms = 0.0;
    double e2e_ms = 0.0;
    double fps = 0.0;
};

class Profiler {
public:
    void AddIteration(const IterationMetrics& metrics);
    int Count() const;
    void ReportToStdout(std::ostream& os) const;
    bool SaveCsv(const std::string& path) const;

private:
    void AddStage(const std::string& name, double value);

    std::vector<IterationMetrics> iterations_;
    std::map<std::string, PercentileMeter> stage_meters_;
    std::map<int, int> actual_batch_size_counts_;
};

}  // namespace edge
