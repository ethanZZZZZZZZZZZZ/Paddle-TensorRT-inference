#pragma once

#include <iosfwd>
#include <string>

namespace edge {

struct InputConfig {
    std::string source_type = "synthetic";
    std::string path = "synthetic://mock";
    int num_streams = 1;
    int num_frames = 3;
    int synthetic_width = 640;
    int synthetic_height = 360;
    int synthetic_channels = 3;
};

struct ModelConfig {
    int input_width = 640;
    int input_height = 640;
    int num_classes = 3;
    int mock_num_boxes = 3;
};

struct PreprocessConfig {
    std::string type = "cpu";
};

struct InferConfig {
    std::string backend = "mock";
    std::string precision = "fp32";
    int batch_size = 1;
    bool enable_dynamic_batch = false;
    int dynamic_batch_timeout_ms = 10;
    int predictor_pool_size = 1;
};

struct CudaConfig {
    int stream_pool_size = 1;
    bool enable_pinned_memory = false;
    bool enable_full_gpu_pipeline = false;
};

struct PaddleConfig {
    std::string model_file;
    std::string params_file;
    std::string model_dir;
    bool use_gpu = false;
    int gpu_mem_mb = 512;
    int gpu_device_id = 0;
    bool enable_ir_optim = true;
    bool enable_memory_optim = true;
};

struct TrtConfig {
    bool enable = false;
    int workspace_size = 1 << 30;
    int max_batch_size = 1;
    int min_subgraph_size = 3;
    std::string precision = "fp32";
    bool use_static = false;
    bool use_calib_mode = false;
    std::string cache_dir;
    std::string int8_calib_images;
    int calib_batch_size = 1;
    int calib_num_batches = 1;
    std::string calib_cache_dir;
    bool enable_dynamic_shape = false;
    std::string dynamic_shape_input_name;
    std::string min_input_shape;
    std::string opt_input_shape;
    std::string max_input_shape;
    bool disable_plugin_fp16 = false;
};

struct TrtAnalysisConfig {
    bool enable = false;
    std::string log_path = "../logs/paddle_trt_analysis.log";
    std::string report_json = "../benchmarks/trt_subgraph_report.json";
    std::string report_md = "../benchmarks/trt_subgraph_report.md";
};

struct PostprocessConfig {
    std::string mode = "mock_yolo";
    std::string decode_backend = "cpu";
    std::string nms_backend = "cpu";
    float score_threshold = 0.25F;
    float nms_threshold = 0.45F;
    int top_k = 100;
    float plugin_int8_input_scale = 1.0F;
};

struct ProfileConfig {
    bool enable_timer = true;
};

struct OutputConfig {
    bool save_result = false;
    std::string result_path = "outputs/mock_result.txt";
};

struct BenchmarkConfig {
    int warmup_iters = 0;
    int benchmark_iters = 3;
    std::string output_csv = "../benchmarks/results.csv";
};

struct AppConfig {
    InputConfig input;
    ModelConfig model;
    PreprocessConfig preprocess;
    InferConfig infer;
    CudaConfig cuda;
    PaddleConfig paddle;
    TrtConfig trt;
    TrtAnalysisConfig trt_analysis;
    PostprocessConfig postprocess;
    ProfileConfig profile;
    OutputConfig output;
    BenchmarkConfig benchmark;
};

class Config {
public:
    static Config LoadFromFile(const std::string& path);

    const AppConfig& Data() const;
    void Print(std::ostream& os) const;

private:
    explicit Config(AppConfig data);

    AppConfig data_;
};

}  // namespace edge
