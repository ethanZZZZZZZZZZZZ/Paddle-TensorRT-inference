#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "common/config.h"
#include "common/types.h"

namespace edge {

class InferEngine;

class PredictorPool {
public:
    struct Request {
        int batch_id = 0;
        TensorBuffer input;
        bool prefer_device_output = false;
    };

    struct Result {
        int batch_id = 0;
        int worker_id = -1;
        bool ok = false;
        std::string error;
        TensorBuffer host_output;
        InferOutput infer_output;
        bool used_infer_output = false;
        double queue_wait_ms = 0.0;
        double inference_ms = 0.0;

    private:
        friend class PredictorPool;

        struct State {
            std::mutex mutex;
            std::condition_variable cv;
            bool released = false;
        };

        std::shared_ptr<State> state;
    };

    PredictorPool() = default;
    ~PredictorPool();

    PredictorPool(const PredictorPool&) = delete;
    PredictorPool& operator=(const PredictorPool&) = delete;

    bool Init(const AppConfig& config);
    bool Submit(Request request);
    bool Pop(Result& result);
    void Release(Result& result);
    void Stop();

    int Size() const;
    std::string Name() const;

private:
    struct Task {
        Request request;
        std::chrono::steady_clock::time_point submit_time;
        std::shared_ptr<Result::State> state;
    };

    struct Worker {
        int id = 0;
        std::unique_ptr<InferEngine> engine;
        std::thread thread;
    };

    std::unique_ptr<InferEngine> CreateEngine(const AppConfig& config) const;
    void WorkerLoop(Worker* worker);
    void NotifyActiveStatesForStop();
    static double ElapsedMs(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end);

    AppConfig config_;
    std::vector<Worker> workers_;
    std::queue<Task> tasks_;
    std::queue<Result> results_;

    mutable std::mutex task_mutex_;
    std::condition_variable task_cv_;
    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    std::mutex active_state_mutex_;
    std::vector<std::weak_ptr<Result::State>> active_states_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<int> outstanding_{0};
};

}  // namespace edge
