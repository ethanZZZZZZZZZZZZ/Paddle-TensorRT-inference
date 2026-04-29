#include "common/config.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace edge {
namespace {

std::string Trim(const std::string& input) {
    const auto begin = std::find_if_not(input.begin(), input.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto end = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();

    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

std::string RemoveComment(const std::string& line) {
    const auto pos = line.find('#');
    if (pos == std::string::npos) {
        return line;
    }
    return line.substr(0, pos);
}

std::string Unquote(std::string value) {
    value = Trim(value);
    if (value.size() >= 2) {
        const char first = value.front();
        const char last = value.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return value.substr(1, value.size() - 2);
        }
    }
    return value;
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

void RequireKey(const std::map<std::string, std::string>& values, const std::string& key) {
    if (values.find(key) == values.end()) {
        throw std::runtime_error("Missing required config key: " + key);
    }
}

int ParseInt(const std::map<std::string, std::string>& values, const std::string& key, int fallback) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return fallback;
    }
    try {
        return std::stoi(it->second);
    } catch (const std::exception& ex) {
        throw std::runtime_error("Invalid integer for config key '" + key + "': " + ex.what());
    }
}

float ParseFloat(const std::map<std::string, std::string>& values, const std::string& key, float fallback) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return fallback;
    }
    try {
        return std::stof(it->second);
    } catch (const std::exception& ex) {
        throw std::runtime_error("Invalid float for config key '" + key + "': " + ex.what());
    }
}

std::string ParseString(
    const std::map<std::string, std::string>& values,
    const std::string& key,
    const std::string& fallback) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return fallback;
    }
    return it->second;
}

bool ParseBool(const std::map<std::string, std::string>& values, const std::string& key, bool fallback) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return fallback;
    }

    const std::string value = ToLower(it->second);
    if (value == "true" || value == "1" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "false" || value == "0" || value == "no" || value == "off") {
        return false;
    }

    throw std::runtime_error("Invalid boolean for config key '" + key + "': " + it->second);
}

bool HasUriScheme(const std::string& path) {
    return path.find("://") != std::string::npos;
}

bool IsAbsolutePath(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    if (path[0] == '/' || path[0] == '\\') {
        return true;
    }
    return path.size() >= 3 &&
           std::isalpha(static_cast<unsigned char>(path[0])) != 0 &&
           path[1] == ':' &&
           (path[2] == '\\' || path[2] == '/');
}

std::string DirName(const std::string& path) {
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return {};
    }
    return path.substr(0, pos);
}

std::string JoinPath(const std::string& lhs, const std::string& rhs) {
    if (lhs.empty() || rhs.empty()) {
        return lhs.empty() ? rhs : lhs;
    }
    const char last = lhs.back();
    if (last == '/' || last == '\\') {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

std::string ResolvePathRelativeToConfig(const std::string& config_path, const std::string& path) {
    if (path.empty() || HasUriScheme(path) || IsAbsolutePath(path)) {
        return path;
    }
    return JoinPath(DirName(config_path), path);
}

bool PathExists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

bool DirectoryExists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

}  // namespace

Config::Config(AppConfig data)
    : data_(std::move(data)) {}

Config Config::LoadFromFile(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    std::map<std::string, std::string> values;
    std::string section;
    std::string line;
    int line_no = 0;

    while (std::getline(input, line)) {
        ++line_no;
        line = Trim(RemoveComment(line));
        if (line.empty()) {
            continue;
        }

        if (line.back() == ':') {
            section = Trim(line.substr(0, line.size() - 1));
            continue;
        }

        const auto colon_pos = line.find(':');
        if (colon_pos == std::string::npos) {
            throw std::runtime_error(
                "Invalid config line " + std::to_string(line_no) + ": expected key: value");
        }

        const std::string key = Trim(line.substr(0, colon_pos));
        const std::string value = Unquote(line.substr(colon_pos + 1));
        if (key.empty()) {
            throw std::runtime_error("Invalid config line " + std::to_string(line_no) + ": empty key");
        }

        const std::string full_key = section.empty() ? key : section + "." + key;
        values[full_key] = value;
    }

    AppConfig config;

    RequireKey(values, "input.source_type");
    RequireKey(values, "input.path");
    RequireKey(values, "input.num_streams");
    RequireKey(values, "model.input_width");
    RequireKey(values, "model.input_height");
    RequireKey(values, "infer.backend");

    config.input.source_type = ToLower(ParseString(values, "input.source_type", config.input.source_type));
    config.input.path = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "input.path", config.input.path));
    config.input.num_streams = ParseInt(values, "input.num_streams", config.input.num_streams);
    config.input.num_frames = ParseInt(values, "input.num_frames", config.input.num_frames);
    config.input.synthetic_width = ParseInt(values, "input.synthetic_width", config.input.synthetic_width);
    config.input.synthetic_height = ParseInt(values, "input.synthetic_height", config.input.synthetic_height);
    config.input.synthetic_channels = ParseInt(values, "input.synthetic_channels", config.input.synthetic_channels);

    config.model.input_width = ParseInt(values, "model.input_width", config.model.input_width);
    config.model.input_height = ParseInt(values, "model.input_height", config.model.input_height);
    config.model.num_classes = ParseInt(values, "model.num_classes", config.model.num_classes);
    config.model.mock_num_boxes = ParseInt(values, "model.mock_num_boxes", config.model.mock_num_boxes);

    config.preprocess.type = ToLower(ParseString(values, "preprocess.type", config.preprocess.type));

    config.infer.backend = ToLower(ParseString(values, "infer.backend", config.infer.backend));
    config.infer.precision = ParseString(values, "infer.precision", config.infer.precision);
    config.infer.batch_size = ParseInt(values, "infer.batch_size", config.infer.batch_size);
    config.infer.enable_dynamic_batch =
        ParseBool(values, "infer.enable_dynamic_batch", config.infer.enable_dynamic_batch);
    config.infer.dynamic_batch_timeout_ms =
        ParseInt(values, "infer.dynamic_batch_timeout_ms", config.infer.dynamic_batch_timeout_ms);
    config.infer.predictor_pool_size =
        ParseInt(values, "infer.predictor_pool_size", config.infer.predictor_pool_size);

    config.cuda.stream_pool_size =
        ParseInt(values, "cuda.stream_pool_size", config.cuda.stream_pool_size);
    config.cuda.enable_pinned_memory =
        ParseBool(values, "cuda.enable_pinned_memory", config.cuda.enable_pinned_memory);

    config.paddle.model_file = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "paddle.model_file", config.paddle.model_file));
    config.paddle.params_file = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "paddle.params_file", config.paddle.params_file));
    config.paddle.model_dir = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "paddle.model_dir", config.paddle.model_dir));
    config.paddle.use_gpu = ParseBool(values, "paddle.use_gpu", config.paddle.use_gpu);
    config.paddle.gpu_mem_mb = ParseInt(values, "paddle.gpu_mem_mb", config.paddle.gpu_mem_mb);
    config.paddle.gpu_device_id = ParseInt(values, "paddle.gpu_device_id", config.paddle.gpu_device_id);
    config.paddle.enable_ir_optim =
        ParseBool(values, "paddle.enable_ir_optim", config.paddle.enable_ir_optim);
    config.paddle.enable_memory_optim =
        ParseBool(values, "paddle.enable_memory_optim", config.paddle.enable_memory_optim);

    config.trt.enable = ParseBool(values, "trt.enable", config.trt.enable);
    config.trt.workspace_size = ParseInt(values, "trt.workspace_size", config.trt.workspace_size);
    config.trt.max_batch_size = ParseInt(values, "trt.max_batch_size", config.infer.batch_size);
    config.trt.min_subgraph_size = ParseInt(values, "trt.min_subgraph_size", config.trt.min_subgraph_size);
    config.trt.precision = ToLower(ParseString(values, "trt.precision", config.infer.precision));
    config.trt.use_static = ParseBool(values, "trt.use_static", config.trt.use_static);
    config.trt.use_calib_mode = ParseBool(values, "trt.use_calib_mode", config.trt.use_calib_mode);
    config.trt.cache_dir = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "trt.cache_dir", config.trt.cache_dir));
    config.trt.int8_calib_images = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "trt.int8_calib_images", config.trt.int8_calib_images));
    config.trt.calib_batch_size = ParseInt(values, "trt.calib_batch_size", config.trt.calib_batch_size);
    config.trt.calib_num_batches = ParseInt(values, "trt.calib_num_batches", config.trt.calib_num_batches);
    config.trt.calib_cache_dir = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "trt.calib_cache_dir", config.trt.calib_cache_dir));
    config.trt.enable_dynamic_shape =
        ParseBool(values, "trt.enable_dynamic_shape", config.trt.enable_dynamic_shape);
    config.trt.dynamic_shape_input_name =
        ParseString(values, "trt.dynamic_shape_input_name", config.trt.dynamic_shape_input_name);
    config.trt.min_input_shape = ParseString(values, "trt.min_input_shape", config.trt.min_input_shape);
    config.trt.opt_input_shape = ParseString(values, "trt.opt_input_shape", config.trt.opt_input_shape);
    config.trt.max_input_shape = ParseString(values, "trt.max_input_shape", config.trt.max_input_shape);
    config.trt.disable_plugin_fp16 =
        ParseBool(values, "trt.disable_plugin_fp16", config.trt.disable_plugin_fp16);

    config.trt_analysis.enable = ParseBool(values, "trt_analysis.enable", config.trt_analysis.enable);
    config.trt_analysis.log_path = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "trt_analysis.log_path", config.trt_analysis.log_path));
    config.trt_analysis.report_json = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "trt_analysis.report_json", config.trt_analysis.report_json));
    config.trt_analysis.report_md = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "trt_analysis.report_md", config.trt_analysis.report_md));

    config.postprocess.mode = ToLower(ParseString(values, "postprocess.mode", config.postprocess.mode));
    config.postprocess.decode_backend =
        ToLower(ParseString(values, "postprocess.decode_backend", config.postprocess.decode_backend));
    config.postprocess.nms_backend =
        ToLower(ParseString(values, "postprocess.nms_backend", config.postprocess.nms_backend));
    config.postprocess.score_threshold =
        ParseFloat(values, "postprocess.score_threshold", config.postprocess.score_threshold);
    config.postprocess.nms_threshold =
        ParseFloat(values, "postprocess.nms_threshold", config.postprocess.nms_threshold);
    config.postprocess.top_k = ParseInt(values, "postprocess.top_k", config.postprocess.top_k);

    config.profile.enable_timer = ParseBool(values, "profile.enable_timer", config.profile.enable_timer);

    config.output.save_result = ParseBool(values, "output.save_result", config.output.save_result);
    config.output.result_path = ParseString(values, "output.result_path", config.output.result_path);

    config.benchmark.warmup_iters =
        ParseInt(values, "benchmark.warmup_iters", config.benchmark.warmup_iters);
    config.benchmark.benchmark_iters =
        ParseInt(values, "benchmark.benchmark_iters", config.benchmark.benchmark_iters);
    config.benchmark.output_csv = ResolvePathRelativeToConfig(
        path,
        ParseString(values, "benchmark.output_csv", config.benchmark.output_csv));

    if (config.trt.precision == "int8" &&
        config.trt.use_calib_mode &&
        config.input.source_type == "image_list" &&
        config.input.num_frames == 0 &&
        config.trt.calib_batch_size > 0 &&
        config.trt.calib_num_batches > 0) {
        config.input.num_frames = config.trt.calib_batch_size * config.trt.calib_num_batches;
    }

    if (config.input.source_type != "synthetic" &&
        config.input.source_type != "video_file" &&
        config.input.source_type != "image_list") {
        throw std::runtime_error("input.source_type must be one of: synthetic, video_file, image_list");
    }
    if (config.input.path.empty()) {
        throw std::runtime_error("input.path must not be empty");
    }
    if (config.input.num_streams <= 0) {
        throw std::runtime_error("input.num_streams must be positive");
    }
#ifndef EDGE_ENABLE_OPENCV
    if (config.input.source_type == "video_file" || config.input.source_type == "image_list") {
        throw std::runtime_error(
            "input.source_type=video_file or image_list requires OpenCV. Reconfigure with -DENABLE_OPENCV=ON "
            "after installing OpenCV, or use input.source_type=synthetic.");
    }
#endif
    if (config.input.source_type == "synthetic" && config.input.num_frames <= 0) {
        throw std::runtime_error("input.num_frames must be positive for synthetic source");
    }
    if ((config.input.source_type == "video_file" || config.input.source_type == "image_list") &&
        config.input.num_frames < 0) {
        throw std::runtime_error(
            "input.num_frames must be >= 0 for video_file/image_list source; 0 means read until EOF");
    }
    if (config.preprocess.type != "cpu" && config.preprocess.type != "gpu") {
        throw std::runtime_error("preprocess.type must be one of: cpu, gpu");
    }
#ifndef EDGE_ENABLE_CUDA
    if (config.preprocess.type == "gpu") {
        throw std::runtime_error(
            "preprocess.type=gpu requires configuring with -DENABLE_CUDA=ON. "
            "Use preprocess.type=cpu for dependency-free runs.");
    }
#endif
    if (config.infer.backend != "mock" && config.infer.backend != "paddle" &&
        config.infer.backend != "paddle_trt") {
        throw std::runtime_error("infer.backend must be one of: mock, paddle, paddle_trt");
    }
#ifndef EDGE_ENABLE_PADDLE
    if (config.infer.backend == "paddle" || config.infer.backend == "paddle_trt") {
        throw std::runtime_error(
            "infer.backend=paddle or paddle_trt requires configuring with -DENABLE_PADDLE=ON and a valid "
            "Paddle Inference installation. Use infer.backend=mock for dependency-free runs.");
    }
#endif
    const std::string precision = ToLower(config.infer.precision);
    if (precision != "fp32" && precision != "fp16" && precision != "int8") {
        throw std::runtime_error("infer.precision must be one of: fp32, fp16, int8");
    }
    config.infer.precision = precision;
    if (config.infer.batch_size <= 0) {
        throw std::runtime_error("infer.batch_size must be positive");
    }
    if (config.infer.dynamic_batch_timeout_ms < 0) {
        throw std::runtime_error("infer.dynamic_batch_timeout_ms must be >= 0");
    }
    if (config.infer.predictor_pool_size <= 0) {
        throw std::runtime_error("infer.predictor_pool_size must be positive");
    }
    if (config.cuda.stream_pool_size <= 0) {
        throw std::runtime_error("cuda.stream_pool_size must be positive");
    }
#if !defined(EDGE_ENABLE_CUDA) && !defined(EDGE_ENABLE_TENSORRT_PLUGIN)
    if (config.cuda.stream_pool_size != 1 || config.cuda.enable_pinned_memory) {
        throw std::runtime_error(
            "cuda.stream_pool_size > 1 or cuda.enable_pinned_memory=true requires "
            "configuring with -DENABLE_CUDA=ON or -DENABLE_TENSORRT_PLUGIN=ON.");
    }
#endif
    if (config.infer.backend == "paddle" || config.infer.backend == "paddle_trt") {
        const bool has_model_dir = !config.paddle.model_dir.empty();
        const bool has_model_pair = !config.paddle.model_file.empty() && !config.paddle.params_file.empty();
        if (has_model_dir == has_model_pair) {
            throw std::runtime_error(
                "paddle backend requires exactly one model source: paddle.model_dir or "
                "paddle.model_file + paddle.params_file");
        }
        if (has_model_dir && !DirectoryExists(config.paddle.model_dir)) {
            throw std::runtime_error(
                "paddle.model_dir does not exist or is not a directory: " + config.paddle.model_dir);
        }
        if (has_model_pair) {
            if (!PathExists(config.paddle.model_file)) {
                throw std::runtime_error("paddle.model_file does not exist: " + config.paddle.model_file);
            }
            if (!PathExists(config.paddle.params_file)) {
                throw std::runtime_error("paddle.params_file does not exist: " + config.paddle.params_file);
            }
        }
        if (config.paddle.gpu_mem_mb <= 0) {
            throw std::runtime_error("paddle.gpu_mem_mb must be positive");
        }
        if (config.paddle.gpu_device_id < 0) {
            throw std::runtime_error("paddle.gpu_device_id must be >= 0");
        }
    }
    if (config.infer.backend == "paddle_trt" && !config.trt.enable) {
        throw std::runtime_error("infer.backend=paddle_trt requires trt.enable: true");
    }
    if (config.trt.enable && config.infer.backend == "mock") {
        throw std::runtime_error("trt.enable=true requires infer.backend: paddle or paddle_trt");
    }
    if (config.trt.enable) {
        if (!config.paddle.use_gpu) {
            throw std::runtime_error("trt.enable=true requires paddle.use_gpu: true");
        }
        if (config.trt.workspace_size <= 0) {
            throw std::runtime_error("trt.workspace_size must be positive");
        }
        if (config.trt.max_batch_size <= 0) {
            throw std::runtime_error("trt.max_batch_size must be positive");
        }
        if (config.trt.max_batch_size < config.infer.batch_size) {
            throw std::runtime_error("trt.max_batch_size must be >= infer.batch_size");
        }
        if (config.trt.min_subgraph_size <= 0) {
            throw std::runtime_error("trt.min_subgraph_size must be positive");
        }
        if (config.trt.precision != "fp32" && config.trt.precision != "fp16" &&
            config.trt.precision != "int8") {
            throw std::runtime_error("trt.precision must be one of: fp32, fp16, int8");
        }
        if (config.trt.precision == "int8" && config.trt.use_calib_mode) {
            if (config.trt.int8_calib_images.empty()) {
                throw std::runtime_error(
                    "trt.precision=int8 with trt.use_calib_mode=true requires trt.int8_calib_images");
            }
            if (!PathExists(config.trt.int8_calib_images)) {
                throw std::runtime_error(
                    "trt.int8_calib_images does not exist: " + config.trt.int8_calib_images);
            }
            if (config.trt.calib_batch_size <= 0) {
                throw std::runtime_error("trt.calib_batch_size must be positive");
            }
            if (config.trt.calib_num_batches <= 0) {
                throw std::runtime_error("trt.calib_num_batches must be positive");
            }
            if (config.trt.calib_cache_dir.empty()) {
                throw std::runtime_error(
                    "trt.precision=int8 with calibration requires trt.calib_cache_dir");
            }
        }
        if (config.trt.precision == "int8" && !config.trt.use_calib_mode) {
            if (config.trt.cache_dir.empty()) {
                throw std::runtime_error(
                    "trt.precision=int8 without calibration requires trt.cache_dir pointing to an existing "
                    "or previously generated TensorRT cache directory");
            }
            if (!DirectoryExists(config.trt.cache_dir)) {
                throw std::runtime_error(
                    "trt.precision=int8 without calibration requires an existing trt.cache_dir: " +
                    config.trt.cache_dir);
            }
        }
        if (config.trt.enable_dynamic_shape) {
            if (config.trt.dynamic_shape_input_name.empty()) {
                throw std::runtime_error(
                    "trt.enable_dynamic_shape=true requires trt.dynamic_shape_input_name");
            }
            if (config.trt.min_input_shape.empty() || config.trt.opt_input_shape.empty() ||
                config.trt.max_input_shape.empty()) {
                throw std::runtime_error(
                    "trt.enable_dynamic_shape=true requires trt.min_input_shape, "
                    "trt.opt_input_shape, and trt.max_input_shape");
            }
        }
    }
    if (config.trt_analysis.enable) {
        if (!config.trt.enable) {
            throw std::runtime_error("trt_analysis.enable=true requires trt.enable: true");
        }
        if (config.trt_analysis.log_path.empty()) {
            throw std::runtime_error("trt_analysis.log_path must not be empty");
        }
        if (config.trt_analysis.report_json.empty()) {
            throw std::runtime_error("trt_analysis.report_json must not be empty");
        }
        if (config.trt_analysis.report_md.empty()) {
            throw std::runtime_error("trt_analysis.report_md must not be empty");
        }
    }
    if (config.model.input_width <= 0 || config.model.input_height <= 0) {
        throw std::runtime_error("model.input_width and model.input_height must be positive");
    }
    if (config.input.synthetic_width <= 0 || config.input.synthetic_height <= 0) {
        throw std::runtime_error("input synthetic frame size must be positive");
    }
    if (config.input.synthetic_channels != 3) {
        throw std::runtime_error("Synthetic source only supports 3 channels");
    }
    if (config.model.num_classes <= 0) {
        throw std::runtime_error("model.num_classes must be positive");
    }
    if (config.model.mock_num_boxes <= 0) {
        throw std::runtime_error("model.mock_num_boxes must be positive");
    }
    if (config.postprocess.mode != "mock_yolo" &&
        config.postprocess.mode != "raw" &&
        config.postprocess.mode != "trt_yolo") {
        throw std::runtime_error("postprocess.mode must be one of: mock_yolo, raw, trt_yolo");
    }
    if (config.postprocess.decode_backend != "cpu" &&
        config.postprocess.decode_backend != "gpu" &&
        config.postprocess.decode_backend != "trt_plugin") {
        throw std::runtime_error("postprocess.decode_backend must be one of: cpu, gpu, trt_plugin");
    }
    if (config.postprocess.nms_backend != "cpu" &&
        config.postprocess.nms_backend != "gpu" &&
        config.postprocess.nms_backend != "trt_plugin") {
        throw std::runtime_error("postprocess.nms_backend must be one of: cpu, gpu, trt_plugin");
    }
    if (config.postprocess.nms_backend == "gpu" && config.postprocess.decode_backend != "gpu") {
        throw std::runtime_error("postprocess.nms_backend=gpu requires postprocess.decode_backend=gpu");
    }
    if (config.postprocess.nms_backend == "trt_plugin" &&
        config.postprocess.decode_backend != "trt_plugin") {
        throw std::runtime_error("postprocess.nms_backend=trt_plugin requires postprocess.decode_backend=trt_plugin");
    }
    if (config.postprocess.decode_backend == "trt_plugin" &&
        config.postprocess.nms_backend != "trt_plugin") {
        throw std::runtime_error("postprocess.decode_backend=trt_plugin requires postprocess.nms_backend=trt_plugin");
    }
#ifndef EDGE_ENABLE_CUDA
    if (config.postprocess.decode_backend == "gpu" || config.postprocess.nms_backend == "gpu") {
        throw std::runtime_error(
            "postprocess.decode_backend=gpu or postprocess.nms_backend=gpu requires "
            "configuring with -DENABLE_CUDA=ON. Use CPU postprocess backends for dependency-free runs.");
    }
#endif
#ifndef EDGE_ENABLE_TENSORRT_PLUGIN
    if (config.postprocess.decode_backend == "trt_plugin" ||
        config.postprocess.nms_backend == "trt_plugin") {
        throw std::runtime_error(
            "postprocess.decode_backend=trt_plugin or postprocess.nms_backend=trt_plugin requires "
            "configuring with -DENABLE_TENSORRT_PLUGIN=ON.");
    }
#endif
    if (config.postprocess.score_threshold < 0.0F || config.postprocess.score_threshold > 1.0F) {
        throw std::runtime_error("postprocess.score_threshold must be in [0, 1]");
    }
    if (config.postprocess.nms_threshold < 0.0F || config.postprocess.nms_threshold > 1.0F) {
        throw std::runtime_error("postprocess.nms_threshold must be in [0, 1]");
    }
    if (config.postprocess.top_k <= 0) {
        throw std::runtime_error("postprocess.top_k must be positive");
    }
    if (config.benchmark.warmup_iters < 0) {
        throw std::runtime_error("benchmark.warmup_iters must be >= 0");
    }
    if (config.benchmark.benchmark_iters < 0) {
        throw std::runtime_error("benchmark.benchmark_iters must be >= 0; 0 means run until sources end");
    }
    if (config.benchmark.output_csv.empty()) {
        throw std::runtime_error("benchmark.output_csv must not be empty");
    }

    return Config(config);
}

const AppConfig& Config::Data() const {
    return data_;
}

void Config::Print(std::ostream& os) const {
    os << "[input]\n";
    os << "source_type: " << data_.input.source_type << '\n';
    os << "path: " << data_.input.path << '\n';
    os << "num_streams: " << data_.input.num_streams << '\n';
    os << "num_frames: " << data_.input.num_frames << '\n';
    os << "synthetic_width: " << data_.input.synthetic_width << '\n';
    os << "synthetic_height: " << data_.input.synthetic_height << '\n';
    os << "synthetic_channels: " << data_.input.synthetic_channels << '\n';

    os << "[model]\n";
    os << "input_width: " << data_.model.input_width << '\n';
    os << "input_height: " << data_.model.input_height << '\n';
    os << "num_classes: " << data_.model.num_classes << '\n';
    os << "mock_num_boxes: " << data_.model.mock_num_boxes << '\n';

    os << "[preprocess]\n";
    os << "type: " << data_.preprocess.type << '\n';

    os << "[infer]\n";
    os << "backend: " << data_.infer.backend << '\n';
    os << "precision: " << data_.infer.precision << '\n';
    os << "batch_size: " << data_.infer.batch_size << '\n';
    os << "enable_dynamic_batch: " << (data_.infer.enable_dynamic_batch ? "true" : "false") << '\n';
    os << "dynamic_batch_timeout_ms: " << data_.infer.dynamic_batch_timeout_ms << '\n';
    os << "predictor_pool_size: " << data_.infer.predictor_pool_size << '\n';

    os << "[cuda]\n";
    os << "stream_pool_size: " << data_.cuda.stream_pool_size << '\n';
    os << "enable_pinned_memory: " << (data_.cuda.enable_pinned_memory ? "true" : "false") << '\n';

    os << "[paddle]\n";
    os << "model_file: " << data_.paddle.model_file << '\n';
    os << "params_file: " << data_.paddle.params_file << '\n';
    os << "model_dir: " << data_.paddle.model_dir << '\n';
    os << "use_gpu: " << (data_.paddle.use_gpu ? "true" : "false") << '\n';
    os << "gpu_mem_mb: " << data_.paddle.gpu_mem_mb << '\n';
    os << "gpu_device_id: " << data_.paddle.gpu_device_id << '\n';
    os << "enable_ir_optim: " << (data_.paddle.enable_ir_optim ? "true" : "false") << '\n';
    os << "enable_memory_optim: " << (data_.paddle.enable_memory_optim ? "true" : "false") << '\n';

    os << "[trt]\n";
    os << "enable: " << (data_.trt.enable ? "true" : "false") << '\n';
    os << "workspace_size: " << data_.trt.workspace_size << '\n';
    os << "max_batch_size: " << data_.trt.max_batch_size << '\n';
    os << "min_subgraph_size: " << data_.trt.min_subgraph_size << '\n';
    os << "precision: " << data_.trt.precision << '\n';
    os << "use_static: " << (data_.trt.use_static ? "true" : "false") << '\n';
    os << "use_calib_mode: " << (data_.trt.use_calib_mode ? "true" : "false") << '\n';
    os << "cache_dir: " << data_.trt.cache_dir << '\n';
    os << "int8_calib_images: " << data_.trt.int8_calib_images << '\n';
    os << "calib_batch_size: " << data_.trt.calib_batch_size << '\n';
    os << "calib_num_batches: " << data_.trt.calib_num_batches << '\n';
    os << "calib_cache_dir: " << data_.trt.calib_cache_dir << '\n';
    os << "enable_dynamic_shape: " << (data_.trt.enable_dynamic_shape ? "true" : "false") << '\n';
    os << "dynamic_shape_input_name: " << data_.trt.dynamic_shape_input_name << '\n';
    os << "min_input_shape: " << data_.trt.min_input_shape << '\n';
    os << "opt_input_shape: " << data_.trt.opt_input_shape << '\n';
    os << "max_input_shape: " << data_.trt.max_input_shape << '\n';
    os << "disable_plugin_fp16: " << (data_.trt.disable_plugin_fp16 ? "true" : "false") << '\n';

    os << "[trt_analysis]\n";
    os << "enable: " << (data_.trt_analysis.enable ? "true" : "false") << '\n';
    os << "log_path: " << data_.trt_analysis.log_path << '\n';
    os << "report_json: " << data_.trt_analysis.report_json << '\n';
    os << "report_md: " << data_.trt_analysis.report_md << '\n';

    os << "[postprocess]\n";
    os << "mode: " << data_.postprocess.mode << '\n';
    os << "decode_backend: " << data_.postprocess.decode_backend << '\n';
    os << "nms_backend: " << data_.postprocess.nms_backend << '\n';
    os << "score_threshold: " << data_.postprocess.score_threshold << '\n';
    os << "nms_threshold: " << data_.postprocess.nms_threshold << '\n';
    os << "top_k: " << data_.postprocess.top_k << '\n';

    os << "[profile]\n";
    os << "enable_timer: " << (data_.profile.enable_timer ? "true" : "false") << '\n';

    os << "[output]\n";
    os << "save_result: " << (data_.output.save_result ? "true" : "false") << '\n';
    os << "result_path: " << data_.output.result_path << '\n';

    os << "[benchmark]\n";
    os << "warmup_iters: " << data_.benchmark.warmup_iters << '\n';
    os << "benchmark_iters: " << data_.benchmark.benchmark_iters << '\n';
    os << "output_csv: " << data_.benchmark.output_csv << '\n';
}

}  // namespace edge
