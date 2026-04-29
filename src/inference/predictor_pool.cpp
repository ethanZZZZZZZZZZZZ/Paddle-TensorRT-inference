#include "inference/predictor_pool.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <sstream>
#include <utility>

#include "common/logging.h"
#include "inference/infer_engine.h"
#include "inference/mock_infer_engine.h"
#ifdef EDGE_ENABLE_PADDLE
#include "inference/paddle_infer_engine.h"
#endif

namespace edge {

PredictorPool::~PredictorPool() {
    Stop();
}

bool PredictorPool::Init(const AppConfig& config) {
    Stop();
    config_ = config;
    stop_requested_.store(false);
    outstanding_.store(0);

    const int pool_size = config_.infer.predictor_pool_size;
    if (pool_size > 1 &&
        config_.infer.backend == "paddle_trt" &&
        config_.trt.use_static &&
        !config_.trt.cache_dir.empty()) {
        EDGE_LOG_WARN("PredictorPool with Paddle-TRT static cache may initialize multiple "
                      "predictors against the same trt.cache_dir. Prefer warming/building "
                      "the cache with predictor_pool_size=1 before enabling the pool.");
    }
    workers_.clear();
    workers_.reserve(static_cast<std::size_t>(pool_size));

    for (int i = 0; i < pool_size; ++i) {
        Worker worker;
        worker.id = i;
        worker.engine = CreateEngine(config_);
        if (!worker.engine) {
            EDGE_LOG_ERROR("PredictorPool failed to create infer engine for worker_id=" << i);
            Stop();
            return false;
        }
        if (!worker.engine->Init(config_)) {
            EDGE_LOG_ERROR("PredictorPool failed to initialize worker_id=" << i
                                                                          << ", engine="
                                                                          << worker.engine->Name());
            Stop();
            return false;
        }
        workers_.push_back(std::move(worker));
    }

    for (auto& worker : workers_) {
        worker.thread = std::thread(&PredictorPool::WorkerLoop, this, &worker);
    }

    EDGE_LOG_INFO("PredictorPool initialized, size=" << workers_.size()
                                                     << ", backend="
                                                     << config_.infer.backend);
    return true;
}

bool PredictorPool::Submit(Request request) {
    if (stop_requested_.load()) {
        EDGE_LOG_ERROR("PredictorPool Submit called after stop");
        return false;
    }

    Task task;
    task.request = std::move(request);
    task.submit_time = std::chrono::steady_clock::now();
    task.state = std::make_shared<Result::State>();
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        tasks_.push(std::move(task));
        ++outstanding_;
    }
    task_cv_.notify_one();
    return true;
}

bool PredictorPool::Pop(Result& result) {
    std::unique_lock<std::mutex> lock(result_mutex_);
    result_cv_.wait(lock, [&]() {
        return !results_.empty() || (stop_requested_.load() && outstanding_.load() == 0);
    });
    if (results_.empty()) {
        return false;
    }

    result = std::move(results_.front());
    results_.pop();
    return true;
}

void PredictorPool::Release(Result& result) {
    if (!result.state) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(result.state->mutex);
        if (!result.state->released) {
            result.state->released = true;
            --outstanding_;
        }
    }
    result.state->cv.notify_all();
    result.state.reset();
}

void PredictorPool::Stop() {
    const bool was_stopped = stop_requested_.exchange(true);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        while (!tasks_.empty()) {
            tasks_.pop();
        }
    }
    task_cv_.notify_all();
    NotifyActiveStatesForStop();
    result_cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.thread.joinable()) {
            worker.thread.join();
        }
    }
    workers_.clear();
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        while (!results_.empty()) {
            results_.pop();
        }
    }
    {
        std::lock_guard<std::mutex> lock(active_state_mutex_);
        active_states_.clear();
    }
    outstanding_.store(0);
    if (!was_stopped) {
        EDGE_LOG_INFO("PredictorPool stopped");
    }
}

int PredictorPool::Size() const {
    return static_cast<int>(workers_.size());
}

std::string PredictorPool::Name() const {
    std::ostringstream oss;
    oss << "PredictorPool(size=" << workers_.size()
        << ", backend=" << config_.infer.backend << ")";
    return oss.str();
}

std::unique_ptr<InferEngine> PredictorPool::CreateEngine(const AppConfig& config) const {
    if (config.infer.backend == "mock") {
        return std::make_unique<MockInferEngine>();
    }
#ifdef EDGE_ENABLE_PADDLE
    if (config.infer.backend == "paddle" || config.infer.backend == "paddle_trt") {
        return std::make_unique<PaddleInferEngine>();
    }
#endif
    EDGE_LOG_ERROR("PredictorPool unsupported infer.backend=" << config.infer.backend);
    return nullptr;
}

void PredictorPool::WorkerLoop(Worker* worker) {
    while (!stop_requested_.load()) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(task_mutex_);
            task_cv_.wait(lock, [&]() {
                return stop_requested_.load() || !tasks_.empty();
            });
            if (stop_requested_.load() && tasks_.empty()) {
                break;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }

        Result result;
        result.batch_id = task.request.batch_id;
        result.worker_id = worker->id;
        result.used_infer_output = task.request.prefer_device_output;
        result.state = task.state;
        result.queue_wait_ms = ElapsedMs(task.submit_time, std::chrono::steady_clock::now());

        {
            std::lock_guard<std::mutex> lock(active_state_mutex_);
            active_states_.push_back(task.state);
        }

        const auto infer_start = std::chrono::steady_clock::now();
        if (task.request.has_device_input) {
            result.used_infer_output = true;
            result.ok = worker->engine->Infer(task.request.device_input, result.infer_output);
        } else if (task.request.prefer_device_output) {
            result.ok = worker->engine->Infer(task.request.input, result.infer_output);
        } else {
            result.ok = worker->engine->Infer(task.request.input, result.host_output);
        }
        result.inference_ms = ElapsedMs(infer_start, std::chrono::steady_clock::now());
        if (!result.ok) {
            result.error = "InferEngine::Infer failed";
        }

        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            results_.push(std::move(result));
        }
        result_cv_.notify_one();

        std::unique_lock<std::mutex> release_lock(task.state->mutex);
        task.state->cv.wait(release_lock, [&]() {
            return task.state->released || stop_requested_.load();
        });

        {
            std::lock_guard<std::mutex> lock(active_state_mutex_);
            active_states_.erase(
                std::remove_if(active_states_.begin(),
                               active_states_.end(),
                               [&](const std::weak_ptr<Result::State>& state) {
                                   auto locked = state.lock();
                                   return !locked || locked == task.state;
                               }),
                active_states_.end());
        }
    }
}

void PredictorPool::NotifyActiveStatesForStop() {
    std::lock_guard<std::mutex> lock(active_state_mutex_);
    for (auto& weak_state : active_states_) {
        if (auto state = weak_state.lock()) {
            state->cv.notify_all();
        }
    }
}

double PredictorPool::ElapsedMs(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return static_cast<double>(us) / 1000.0;
}

}  // namespace edge
