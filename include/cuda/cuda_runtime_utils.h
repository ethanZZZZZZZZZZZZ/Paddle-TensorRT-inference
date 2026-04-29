#pragma once

#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "common/types.h"

namespace edge {

class CudaStreamLease {
public:
    CudaStreamLease() = default;
    CudaStreamLease(cudaStream_t stream, int index);

    cudaStream_t Get() const;
    int Index() const;
    bool Valid() const;

private:
    cudaStream_t stream_ = nullptr;
    int index_ = -1;
};

class CudaStreamPool {
public:
    CudaStreamPool() = default;
    ~CudaStreamPool();

    CudaStreamPool(const CudaStreamPool&) = delete;
    CudaStreamPool& operator=(const CudaStreamPool&) = delete;

    bool Init(int stream_count);
    CudaStreamLease Acquire();
    void Reset();
    int Size() const;
    std::string Name() const;

private:
    std::vector<cudaStream_t> streams_;
    std::atomic<int> next_{0};
};

template <typename T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;
    ~PinnedHostBuffer() {
        Reset();
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    bool Allocate(std::size_t elements) {
        if (elements_ == elements && data_ != nullptr) {
            return true;
        }
        Reset();
        if (elements == 0) {
            return true;
        }
        void* raw = nullptr;
        const cudaError_t status = cudaHostAlloc(&raw, elements * sizeof(T), cudaHostAllocDefault);
        if (status != cudaSuccess) {
            return false;
        }
        data_ = static_cast<T*>(raw);
        elements_ = elements;
        return true;
    }

    void Reset() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
            data_ = nullptr;
        }
        elements_ = 0;
    }

    T* Data() {
        return data_;
    }

    const T* Data() const {
        return data_;
    }

    std::size_t Elements() const {
        return elements_;
    }

    std::size_t Bytes() const {
        return elements_ * sizeof(T);
    }

private:
    T* data_ = nullptr;
    std::size_t elements_ = 0;
};

bool AllocatePinnedFloatTensor(
    const std::vector<int64_t>& shape,
    std::size_t elements,
    TensorBuffer& tensor);

}  // namespace edge

#endif  // defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)
